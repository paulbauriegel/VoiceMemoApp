package com.example.voicememo.tools;

import com.google.flatbuffers.Table;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.BuiltinOperator;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.Operator;
import org.tensorflow.lite.schema.OperatorCode;
import org.tensorflow.lite.schema.QuantizationParameters;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorType;

/**
 * Creates a narrow FLOAT32-weight control from the pinned two-decoder-block
 * boundary model.
 *
 * <p>Only blockwise INT4 constants consumed by the executed 156-operator
 * prefix are expanded. The dequantized value is written as FLOAT32, so the
 * control removes LiteRT's blockwise INT4 runtime path without introducing a
 * second weight-rounding step. The production model is never modified.
 */
public final class WhisperFloatDecoderProbeTool {
  private static final long SOURCE_SIZE = 755_273_920L;
  private static final String SOURCE_SHA256 =
      "f5926743061104e49dd12f532fe3d35a3025f791bc1ec75c4e2a0b7eef136337";
  private static final int DECODE_SUBGRAPH = 0;
  private static final int PREFIX_OPERATOR_COUNT = 156;
  private static final int INT4_TENSOR_TYPE = 17;
  private static final int BLOCKWISE_QUANTIZATION_DETAILS = 2;
  private static final int EXPECTED_BLOCK_SIZE = 64;
  private static final int FILE_ALIGNMENT = 16;
  private static final int OUTPUT_BUFFER_BYTES = 1024 * 1024;

  private static final Field TABLE_POSITION;
  private static final Method TABLE_OFFSET;

  static {
    try {
      TABLE_POSITION = Table.class.getDeclaredField("bb_pos");
      TABLE_POSITION.setAccessible(true);
      TABLE_OFFSET = Table.class.getDeclaredMethod("__offset", int.class);
      TABLE_OFFSET.setAccessible(true);
    } catch (ReflectiveOperationException error) {
      throw new ExceptionInInitializerError(error);
    }
  }

  private WhisperFloatDecoderProbeTool() {}

  public static void main(String[] args) {
    try {
      if (args.length == 1 && "self-test".equals(args[0])) {
        selfTest();
        return;
      }
      require(args.length == 3 && "generate".equals(args[0]),
          "Usage: WhisperFloatDecoderProbeTool generate SOURCE OUTPUT");
      generate(Path.of(args[1]), Path.of(args[2]));
    } catch (ToolFailure error) {
      System.err.println("Whisper float decoder probe failed: " + error.getMessage());
      System.exit(2);
    } catch (Exception error) {
      System.err.println(
          "Whisper float decoder probe failed: "
              + error.getClass().getSimpleName() + ": " + error.getMessage());
      System.exit(3);
    }
  }

  private static void generate(Path sourceArgument, Path outputArgument)
      throws IOException {
    Path source = sourceArgument.toAbsolutePath().normalize();
    Path output = outputArgument.toAbsolutePath().normalize();
    require(Files.isRegularFile(source), "Source boundary model is missing: " + source);
    require(Files.size(source) == SOURCE_SIZE,
        "Source boundary model size changed; expected " + SOURCE_SIZE + " bytes.");
    require(SOURCE_SHA256.equals(sha256(source)),
        "Source boundary model SHA-256 does not match the pinned artifact.");
    require(!source.equals(output), "Output must not replace the source model.");
    require(hasBuildPathSegment(output),
        "Diagnostic output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      ConversionPlan plan;
      try (FileChannel sourceChannel =
               FileChannel.open(source, StandardOpenOption.READ)) {
        ByteBuffer sourceBytes =
            sourceChannel.map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes, "Pinned two-block boundary");
        plan = inspectConversionPlan(sourceModel, sourceBytes.limit());

        try (FileChannel outputChannel =
                 FileChannel.open(
                     temporary,
                     StandardOpenOption.READ,
                     StandardOpenOption.WRITE)) {
          long appendOffset = align(outputChannel.size(), FILE_ALIGNMENT);
          if (appendOffset > outputChannel.size()) {
            writeFully(
                outputChannel,
                ByteBuffer.allocate(Math.toIntExact(appendOffset - outputChannel.size())),
                outputChannel.size());
          }

          // Keep only the executable two-block prefix and its decode signature.
          // This prevents delegate preparation from inspecting the diagnostic
          // tail subgraph, whose remaining constants intentionally stay INT4.
          writeInt(outputChannel, plan.subgraphVectorLengthField(), 1);
          writeInt(outputChannel, plan.signatureVectorLengthField(), 1);

          for (WeightPatch patch : plan.patches()) {
            appendOffset = align(appendOffset, FILE_ALIGNMENT);
            long floatBytes = Math.multiplyExact(patch.elementCount(), Float.BYTES);
            dequantizeToFloat32(
                sourceBytes,
                patch,
                outputChannel,
                appendOffset);
            writeLong(
                outputChannel,
                patch.bufferOffsetField(),
                appendOffset);
            writeLong(
                outputChannel,
                patch.bufferSizeField(),
                floatBytes);
            writeByte(
                outputChannel,
                patch.tensorTypeField(),
                TensorType.FLOAT32);
            writeByte(
                outputChannel,
                patch.quantizationDetailsTypeField(),
                (byte) 0);
            appendOffset = Math.addExact(appendOffset, floatBytes);
          }
          outputChannel.force(true);
        }
      }

      moveAtomically(temporary, output);
      verifyOutput(output, plan.patches());
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("dequantized_weight_tensors=" + plan.patches().size());
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static ConversionPlan inspectConversionPlan(
      Model model,
      int sourceSize) {
    require(model.subgraphsLength() >= 1, "Decode subgraph is missing.");
    SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
    require(decode.operatorsLength() >= PREFIX_OPERATOR_COUNT,
        "Decode graph has fewer than " + PREFIX_OPERATOR_COUNT + " operators.");

    List<WeightPatch> patches = new ArrayList<>();
    Set<Integer> seenTensors = new HashSet<>();
    Set<Long> seenBuffers = new HashSet<>();
    for (int operatorIndex = 0;
        operatorIndex < PREFIX_OPERATOR_COUNT;
        operatorIndex++) {
      Operator operator = decode.operators(operatorIndex);
      OperatorCode code = model.operatorCodes((int) operator.opcodeIndex());
      if (code.builtinCode() != BuiltinOperator.FULLY_CONNECTED) continue;
      require(operator.inputsLength() >= 2,
          "FULLY_CONNECTED " + operatorIndex + " has no weight input.");
      int tensorIndex = operator.inputs(1);
      require(tensorIndex >= 0,
          "FULLY_CONNECTED " + operatorIndex + " has an optional weight.");
      Tensor weight = decode.tensors(tensorIndex);
      if ((weight.type() & 0xff) != INT4_TENSOR_TYPE) continue;
      if (!seenTensors.add(tensorIndex)) continue;

      long bufferIndex = weight.buffer();
      require(seenBuffers.add(bufferIndex),
          "Two active INT4 tensors unexpectedly share buffer " + bufferIndex + ".");
      Buffer weightBuffer = model.buffers(Math.toIntExact(bufferIndex));
      require(weightBuffer.dataLength() == 0,
          "INT4 weight " + tensorIndex + " must use an external buffer.");
      long weightOffset = tableLong(weightBuffer, 6);
      long weightBytes = tableLong(weightBuffer, 8);
      long elementCount = tensorElementCount(weight);
      require(weightBytes == (elementCount + 1L) / 2L,
          "INT4 weight " + tensorIndex + " packed byte count changed.");
      requireRegion(weightOffset, weightBytes, sourceSize,
          "INT4 weight " + tensorIndex);

      QuantizationParameters quantization = weight.quantization();
      require(quantization != null,
          "INT4 weight " + tensorIndex + " has no quantization metadata.");
      require((quantization.detailsType() & 0xff)
              == BLOCKWISE_QUANTIZATION_DETAILS,
          "INT4 weight " + tensorIndex + " is not blockwise quantized.");
      Table blockwise = quantization.details(new Table());
      require(blockwise != null,
          "INT4 weight " + tensorIndex + " blockwise details are missing.");
      int scaleTensorIndex = tableInt(blockwise, 4, -1);
      int zeroPointTensorIndex = tableInt(blockwise, 6, -1);
      int blockSize = tableInt(blockwise, 8, -1);
      require(zeroPointTensorIndex == -1,
          "INT4 weight " + tensorIndex + " unexpectedly uses zero points.");
      require(blockSize == EXPECTED_BLOCK_SIZE,
          "INT4 weight " + tensorIndex + " block size changed.");
      require(scaleTensorIndex >= 0 && scaleTensorIndex < decode.tensorsLength(),
          "INT4 weight " + tensorIndex + " scale tensor is invalid.");

      Tensor scale = decode.tensors(scaleTensorIndex);
      require(scale.type() == TensorType.FLOAT16,
          "INT4 weight " + tensorIndex + " scale tensor is not FLOAT16.");
      long scaleCount = tensorElementCount(scale);
      long expectedScaleCount =
          (elementCount + EXPECTED_BLOCK_SIZE - 1L) / EXPECTED_BLOCK_SIZE;
      require(scaleCount == expectedScaleCount,
          "INT4 weight " + tensorIndex + " scale count changed.");
      Buffer scaleBuffer = model.buffers(Math.toIntExact(scale.buffer()));
      require(scaleBuffer.dataLength() == 0,
          "INT4 weight " + tensorIndex + " scale must be external.");
      long scaleOffset = tableLong(scaleBuffer, 6);
      long scaleBytes = tableLong(scaleBuffer, 8);
      require(scaleBytes == scaleCount * Short.BYTES,
          "INT4 weight " + tensorIndex + " scale byte count changed.");
      requireRegion(scaleOffset, scaleBytes, sourceSize,
          "INT4 weight " + tensorIndex + " scales");

      patches.add(
          new WeightPatch(
              operatorIndex,
              tensorIndex,
              elementCount,
              weightOffset,
              scaleOffset,
              blockSize,
              tableFieldPosition(weightBuffer, 6,
                  "weight buffer offset"),
              tableFieldPosition(weightBuffer, 8,
                  "weight buffer size"),
              tableFieldPosition(weight, 6,
                  "weight tensor type"),
              tableFieldPosition(quantization, 12,
                  "quantization details type")));
    }
    require(!patches.isEmpty(),
        "No blockwise INT4 FULLY_CONNECTED weights were found in the prefix.");
    int subgraphVector = tableVectorStart(model, 8, "model subgraphs");
    int signatureVector = tableVectorStart(model, 18, "model signatures");
    require(model.subgraphsLength() == 2,
        "Boundary source must contain the prefix and tail subgraphs.");
    require(model.signatureDefsLength() == 2,
        "Boundary source must contain the prefix and tail signatures.");
    return new ConversionPlan(patches, subgraphVector, signatureVector);
  }

  private static void dequantizeToFloat32(
      ByteBuffer source,
      WeightPatch patch,
      FileChannel output,
      long outputOffset) throws IOException {
    ByteBuffer destination =
        ByteBuffer.allocateDirect(OUTPUT_BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
    long writeOffset = outputOffset;
    for (long element = 0; element < patch.elementCount(); element++) {
      if (destination.remaining() < Float.BYTES) {
        destination.flip();
        int bytes = destination.remaining();
        writeFully(output, destination, writeOffset);
        writeOffset += bytes;
        destination.clear();
      }
      int packed = source.get(
          Math.toIntExact(patch.weightOffset() + element / 2L)) & 0xff;
      int nibble = (element & 1L) == 0L ? packed & 0x0f : (packed >>> 4) & 0x0f;
      int signedValue = nibble >= 8 ? nibble - 16 : nibble;
      int scaleBits =
          source.getShort(
              Math.toIntExact(
                  patch.scaleOffset()
                      + (element / patch.blockSize()) * Short.BYTES)) & 0xffff;
      float scale = halfToFloat(scaleBits);
      require(Float.isFinite(scale),
          "Non-finite scale in tensor " + patch.tensorIndex() + ".");
      destination.putFloat(signedValue * scale);
    }
    destination.flip();
    int remaining = destination.remaining();
    writeFully(output, destination, writeOffset);
    writeOffset += remaining;
    require(writeOffset
            == outputOffset + patch.elementCount() * Float.BYTES,
        "FLOAT32 output size mismatch for tensor " + patch.tensorIndex() + ".");
  }

  private static void verifyOutput(
      Path output,
      List<WeightPatch> sourcePatches) throws IOException {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      require(channel.size() > SOURCE_SIZE,
          "FLOAT32 control did not grow beyond the source model.");
      ByteBuffer bytes =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes, "Generated FLOAT32 control");
      require(model.subgraphsLength() == 1,
          "Generated control must expose only the prefix subgraph.");
      require(model.signatureDefsLength() == 1,
          "Generated control must expose only the decode signature.");
      SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
      for (WeightPatch sourcePatch : sourcePatches) {
        Tensor weight = decode.tensors(sourcePatch.tensorIndex());
        require(weight.type() == TensorType.FLOAT32,
            "Generated tensor " + sourcePatch.tensorIndex()
                + " is not FLOAT32.");
        Buffer buffer = model.buffers(Math.toIntExact(weight.buffer()));
        require(tableLong(buffer, 8)
                == sourcePatch.elementCount() * Float.BYTES,
            "Generated tensor " + sourcePatch.tensorIndex()
                + " FLOAT32 byte count changed.");
        QuantizationParameters quantization = weight.quantization();
        require(quantization != null && quantization.detailsType() == 0,
            "Generated tensor " + sourcePatch.tensorIndex()
                + " still activates blockwise quantization.");
      }

      for (int operatorIndex = 0;
          operatorIndex < PREFIX_OPERATOR_COUNT;
          operatorIndex++) {
        Operator operator = decode.operators(operatorIndex);
        OperatorCode code = model.operatorCodes((int) operator.opcodeIndex());
        if (code.builtinCode() != BuiltinOperator.FULLY_CONNECTED
            || operator.inputsLength() < 2
            || operator.inputs(1) < 0) {
          continue;
        }
        Tensor weight = decode.tensors(operator.inputs(1));
        require((weight.type() & 0xff) != INT4_TENSOR_TYPE,
            "Active FULLY_CONNECTED " + operatorIndex
                + " still has an INT4 weight.");
      }
    }
  }

  private static Model parseModel(ByteBuffer bytes, String label) {
    require(Model.ModelBufferHasIdentifier(bytes.duplicate()),
        label + " does not have the TFL3 identifier.");
    Model model = Model.getRootAsModel(bytes);
    require(model.version() == 3, label + " schema version is not 3.");
    return model;
  }

  private static long tensorElementCount(Tensor tensor) {
    require(tensor.shapeLength() > 0,
        "Tensor " + tensor.name() + " has no shape.");
    long result = 1;
    for (int index = 0; index < tensor.shapeLength(); index++) {
      require(tensor.shape(index) > 0,
          "Tensor " + tensor.name() + " has a non-positive dimension.");
      result = Math.multiplyExact(result, tensor.shape(index));
    }
    return result;
  }

  private static float halfToFloat(int half) {
    int sign = (half & 0x8000) << 16;
    int exponent = (half >>> 10) & 0x1f;
    int mantissa = half & 0x03ff;
    int floatBits;
    if (exponent == 0) {
      if (mantissa == 0) {
        floatBits = sign;
      } else {
        int normalizedExponent = -14;
        while ((mantissa & 0x0400) == 0) {
          mantissa <<= 1;
          normalizedExponent--;
        }
        mantissa &= 0x03ff;
        floatBits =
            sign
                | ((normalizedExponent + 127) << 23)
                | (mantissa << 13);
      }
    } else if (exponent == 0x1f) {
      floatBits = sign | 0x7f800000 | (mantissa << 13);
    } else {
      floatBits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return Float.intBitsToFloat(floatBits);
  }

  private static void selfTest() {
    require(halfToFloat(0x0000) == 0.0f, "FP16 positive zero conversion failed.");
    require(
        Float.floatToRawIntBits(halfToFloat(0x8000))
            == Float.floatToRawIntBits(-0.0f),
        "FP16 negative zero conversion failed.");
    require(halfToFloat(0x3c00) == 1.0f, "FP16 one conversion failed.");
    require(halfToFloat(0xc000) == -2.0f, "FP16 minus two conversion failed.");
    require(halfToFloat(0x0001) == Math.scalb(1.0f, -24),
        "FP16 subnormal conversion failed.");
    require(Float.isInfinite(halfToFloat(0x7c00)),
        "FP16 infinity conversion failed.");
    System.out.println("Whisper float decoder probe self-test passed.");
  }

  private static boolean hasBuildPathSegment(Path path) {
    for (Path segment : path) {
      if ("build".equals(segment.toString())) return true;
    }
    return false;
  }

  private static void requireRegion(
      long offset,
      long size,
      long fileSize,
      String label) {
    require(offset >= 0 && size >= 0 && offset <= fileSize - size,
        label + " is outside the source model.");
  }

  private static long align(long value, int alignment) {
    long remainder = value % alignment;
    return remainder == 0 ? value : Math.addExact(value, alignment - remainder);
  }

  private static int tablePosition(Table table) {
    try {
      return (int) TABLE_POSITION.get(table);
    } catch (ReflectiveOperationException error) {
      throw new ToolFailure("Could not inspect FlatBuffer table position.", error);
    }
  }

  private static int tableOffset(Table table, int vtableOffset) {
    try {
      return (int) TABLE_OFFSET.invoke(table, vtableOffset);
    } catch (ReflectiveOperationException error) {
      throw new ToolFailure("Could not inspect FlatBuffer field.", error);
    }
  }

  private static int tableFieldPosition(
      Table table,
      int vtableOffset,
      String label) {
    int offset = tableOffset(table, vtableOffset);
    require(offset != 0, label + " field is missing.");
    return tablePosition(table) + offset;
  }

  private static int tableVectorStart(
      Table table,
      int vtableOffset,
      String label) {
    int field = tableFieldPosition(table, vtableOffset, label);
    int relative = table.getByteBuffer().getInt(field);
    require(relative > 0, label + " vector offset is invalid.");
    int vector = field + relative;
    require(vector >= 0
            && vector <= table.getByteBuffer().limit() - Integer.BYTES,
        label + " vector is outside the FlatBuffer.");
    return vector;
  }

  private static long tableLong(Table table, int vtableOffset) {
    int offset = tableOffset(table, vtableOffset);
    if (offset == 0) return 0;
    return table.getByteBuffer().getLong(tablePosition(table) + offset);
  }

  private static int tableInt(
      Table table,
      int vtableOffset,
      int defaultValue) {
    int offset = tableOffset(table, vtableOffset);
    if (offset == 0) return defaultValue;
    return table.getByteBuffer().getInt(tablePosition(table) + offset);
  }

  private static void writeByte(
      FileChannel channel,
      long offset,
      byte value) throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(1);
    bytes.put(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeLong(
      FileChannel channel,
      long offset,
      long value) throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putLong(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeInt(
      FileChannel channel,
      long offset,
      int value) throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeFully(
      FileChannel channel,
      ByteBuffer source,
      long offset) throws IOException {
    long position = offset;
    while (source.hasRemaining()) {
      int written = channel.write(source, position);
      require(written > 0, "Could not write output model at byte " + position + ".");
      position += written;
    }
  }

  private static String sha256(Path path) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (Exception error) {
      throw new ToolFailure("SHA-256 is unavailable.", error);
    }
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer bytes = ByteBuffer.allocateDirect(1024 * 1024);
      while (channel.read(bytes) >= 0) {
        bytes.flip();
        digest.update(bytes);
        bytes.clear();
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void moveAtomically(Path source, Path destination)
      throws IOException {
    try {
      Files.move(
          source,
          destination,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING);
    } catch (AtomicMoveNotSupportedException ignored) {
      Files.move(source, destination, StandardCopyOption.REPLACE_EXISTING);
    }
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new ToolFailure(message);
  }

  private record WeightPatch(
      int operatorIndex,
      int tensorIndex,
      long elementCount,
      long weightOffset,
      long scaleOffset,
      int blockSize,
      int bufferOffsetField,
      int bufferSizeField,
      int tensorTypeField,
      int quantizationDetailsTypeField) {}

  private record ConversionPlan(
      List<WeightPatch> patches,
      int subgraphVectorLengthField,
      int signatureVectorLengthField) {}

  private static final class ToolFailure extends RuntimeException {
    ToolFailure(String message) {
      super(message);
    }

    ToolFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
