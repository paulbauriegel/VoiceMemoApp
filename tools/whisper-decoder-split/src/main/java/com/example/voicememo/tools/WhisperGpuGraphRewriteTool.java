package com.example.voicememo.tools;

import com.google.flatbuffers.FlatBufferBuilder;
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
import java.util.HexFormat;
import java.util.List;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.BuiltinOperator;
import org.tensorflow.lite.schema.FullyConnectedOptions;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.Operator;
import org.tensorflow.lite.schema.OperatorCode;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorType;

/**
 * Creates an isolated Whisper decoder with GPU-compatible residual shapes.
 *
 * <p>The original public token contract remains {@code [1, 128]}; GATHER stays
 * on CPU because LiteRT's OpenCL parser cannot consume Whisper's constant
 * embedding table. The position tensor and four projection/residual paths are
 * re-ranked from {@code [128, 1280]} to {@code [1, 128, 1280]} without changing
 * element order. This removes the broadcast ADD barriers so the remaining 155
 * operations can form one GPU partition. The production app model is never
 * modified.
 */
public final class WhisperGpuGraphRewriteTool {
  private static final long SOURCE_SIZE = 964_989_120L;
  private static final String SOURCE_SHA256 =
      "0dd5a7645060263c8b10a88f10e99b991b2ab592cfe098d509945fdd6331a08d";
  private static final int DECODE_SUBGRAPH = 0;
  private static final int EXPECTED_OPERATOR_COUNT = 156;
  private static final int TARGET_LENGTH = 128;
  private static final int HIDDEN_WIDTH = 1280;
  private static final int[] PROJECTION_RESHAPE_OPERATORS = {39, 62, 116, 139};
  private static final int[] PROJECTION_OPERATORS = {40, 63, 117, 140};

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

  private WhisperGpuGraphRewriteTool() {}

  public static void main(String[] args) {
    try {
      require(
          args.length == 3 && "generate".equals(args[0]),
          "Usage: WhisperGpuGraphRewriteTool generate SOURCE OUTPUT");
      generate(Path.of(args[1]), Path.of(args[2]));
    } catch (ToolFailure error) {
      System.err.println("Whisper GPU graph rewrite failed: " + error.getMessage());
      System.exit(2);
    } catch (Exception error) {
      System.err.println(
          "Whisper GPU graph rewrite failed: "
              + error.getClass().getSimpleName()
              + ": "
              + error.getMessage());
      System.exit(3);
    }
  }

  private static void generate(Path sourceArgument, Path outputArgument)
      throws IOException {
    Path source = sourceArgument.toAbsolutePath().normalize();
    Path output = outputArgument.toAbsolutePath().normalize();
    require(Files.isRegularFile(source), "Source FLOAT32 model is missing: " + source);
    require(
        Files.size(source) == SOURCE_SIZE,
        "Source FLOAT32 model size changed; expected " + SOURCE_SIZE + " bytes.");
    require(
        SOURCE_SHA256.equals(sha256(source)),
        "Source FLOAT32 model SHA-256 does not match the pinned artifact.");
    require(!source.equals(output), "Output must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "Diagnostic output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      RewritePlan plan;
      try (FileChannel sourceChannel =
          FileChannel.open(source, StandardOpenOption.READ)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        plan = inspectPlan(parseModel(sourceBytes, "Pinned FLOAT32 decoder"));
      }

      try (FileChannel outputChannel =
          FileChannel.open(
              temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        long appendOffset = align(outputChannel.size(), Integer.BYTES);
        int rankedShapeVector =
            appendIntVector(
                outputChannel,
                appendOffset,
                new int[] {1, TARGET_LENGTH, HIDDEN_WIDTH});
        appendOffset =
            align(
                appendOffset + Integer.BYTES * 4L,
                Integer.BYTES);

        int shapeTensorShapeVector =
            appendIntVector(outputChannel, appendOffset, new int[] {3});
        appendOffset =
            align(
                appendOffset + Integer.BYTES * 2L,
                Integer.BYTES);

        int shapeTensorDataVector =
            appendByteVector(
                outputChannel,
                appendOffset,
                intBytes(1, TARGET_LENGTH, HIDDEN_WIDTH));
        appendOffset =
            align(
                appendOffset + Integer.BYTES + 3L * Integer.BYTES,
                Integer.BYTES);

        ByteBuffer optionsFragment =
            buildFullyConnectedOptions(plan.fullyConnectedOptions());
        int fragmentLength = optionsFragment.remaining();
        int fragmentRoot = optionsFragment.getInt(optionsFragment.position());
        require(
            fragmentRoot > 0 && fragmentRoot < fragmentLength,
            "Fully-connected options fragment root is invalid.");
        writeFully(outputChannel, optionsFragment, appendOffset);
        int fullyConnectedOptionsTarget =
            Math.toIntExact(appendOffset + fragmentRoot);

        for (int shapeField : plan.rankedTensorShapeFields()) {
          writeForwardOffset(
              outputChannel, shapeField, rankedShapeVector, "ranked tensor shape");
        }
        writeForwardOffset(
            outputChannel,
            plan.shapeTensorShapeField(),
            shapeTensorShapeVector,
            "reshape-constant tensor shape");
        writeForwardOffset(
            outputChannel,
            plan.shapeTensorDataField(),
            shapeTensorDataVector,
            "reshape-constant data");
        for (int optionsField : plan.fullyConnectedOptionsFields()) {
          writeForwardOffset(
              outputChannel,
              optionsField,
              fullyConnectedOptionsTarget,
              "fully-connected options");
        }
        outputChannel.force(true);
      }

      moveAtomically(temporary, output);
      verifyOutput(output);
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("tokens_shape=[1,128]");
      System.out.println("aligned_residual_shape=[1,128,1280]");
      System.out.println("cpu_gather_gpu_candidate_ops=155");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static RewritePlan inspectPlan(Model model) {
    require(model.subgraphsLength() == 1, "Expected one decoder subgraph.");
    SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
    require(
        decode.operatorsLength() == EXPECTED_OPERATOR_COUNT,
        "Decoder operator count changed.");
    require(decode.inputsLength() == 3, "Decoder input count changed.");

    Operator gather = requireOperator(model, decode, 0, BuiltinOperator.GATHER);
    require(gather.inputsLength() == 2, "Decoder GATHER input count changed.");
    Tensor tokens = decode.tensors(gather.inputs(1));
    require(tokens.type() == TensorType.INT32, "Token input is not INT32.");
    requireShape(tokens, 1, TARGET_LENGTH);
    require(
        containsTensor(decode, true, gather.inputs(1)),
        "Tokens are not a graph input.");
    requireShape(
        decode.tensors(gather.inputs(0)), 51_866, HIDDEN_WIDTH);
    requireShape(
        decode.tensors(gather.outputs(0)), 1, TARGET_LENGTH, HIDDEN_WIDTH);

    Operator positionAdd = requireOperator(model, decode, 1, BuiltinOperator.ADD);
    require(
        positionAdd.inputs(0) == gather.outputs(0),
        "Position ADD no longer consumes the embedding GATHER.");
    Tensor position = decode.tensors(positionAdd.inputs(1));
    requireShape(position, TARGET_LENGTH, HIDDEN_WIDTH);

    List<Integer> rankedTensorShapeFields = new ArrayList<>();
    rankedTensorShapeFields.add(
        tableFieldPosition(position, 4, "position tensor shape"));
    Tensor reshapeShape = null;
    List<Integer> optionsFields = new ArrayList<>();
    FullyConnectedOptions commonOptions = null;
    for (int index = 0; index < PROJECTION_RESHAPE_OPERATORS.length; index++) {
      Operator reshape =
          requireOperator(
              model,
              decode,
              PROJECTION_RESHAPE_OPERATORS[index],
              BuiltinOperator.RESHAPE);
      require(reshape.inputsLength() == 2, "Projection RESHAPE input count changed.");
      Tensor currentShape = decode.tensors(reshape.inputs(1));
      if (reshapeShape == null) {
        reshapeShape = currentShape;
      } else {
        require(
            reshape.inputs(1)
                == decode
                    .operators(PROJECTION_RESHAPE_OPERATORS[0])
                    .inputs(1),
            "Projection RESHAPE operations no longer share one shape tensor.");
      }
      requireShape(currentShape, 2);
      requireIntTensorData(model, currentShape, TARGET_LENGTH, HIDDEN_WIDTH);

      Tensor reshapeOutput = decode.tensors(reshape.outputs(0));
      requireShape(reshapeOutput, TARGET_LENGTH, HIDDEN_WIDTH);
      rankedTensorShapeFields.add(
          tableFieldPosition(reshapeOutput, 4, "projection reshape output shape"));

      Operator projection =
          requireOperator(
              model,
              decode,
              PROJECTION_OPERATORS[index],
              BuiltinOperator.FULLY_CONNECTED);
      require(
          projection.inputs(0) == reshape.outputs(0),
          "Projection no longer consumes its RESHAPE output.");
      Tensor projectionOutput = decode.tensors(projection.outputs(0));
      requireShape(projectionOutput, TARGET_LENGTH, HIDDEN_WIDTH);
      rankedTensorShapeFields.add(
          tableFieldPosition(
              projectionOutput, 4, "fully-connected projection output shape"));

      FullyConnectedOptions options =
          (FullyConnectedOptions)
              projection.builtinOptions(new FullyConnectedOptions());
      require(options != null, "Projection options are missing.");
      require(!options.keepNumDims(), "Projection already keeps input rank.");
      if (commonOptions == null) {
        commonOptions = options;
      } else {
        require(
            options.fusedActivationFunction()
                    == commonOptions.fusedActivationFunction()
                && options.weightsFormat() == commonOptions.weightsFormat()
                && options.asymmetricQuantizeInputs()
                    == commonOptions.asymmetricQuantizeInputs(),
            "Projection fully-connected options differ.");
      }
      optionsFields.add(
          tableFieldPosition(projection, 12, "fully-connected options"));
    }
    require(reshapeShape != null, "Projection reshape constant is missing.");
    Buffer reshapeShapeBuffer =
        model.buffers(Math.toIntExact(reshapeShape.buffer()));
    require(
        reshapeShapeBuffer.dataLength() == 2 * Integer.BYTES,
        "Projection reshape constant is not inline INT32[2].");
    return new RewritePlan(
        rankedTensorShapeFields,
        tableFieldPosition(reshapeShape, 4, "reshape-constant tensor shape"),
        tableFieldPosition(reshapeShapeBuffer, 4, "reshape-constant data"),
        optionsFields,
        commonOptions);
  }

  private static void verifyOutput(Path output) throws IOException {
    require(Files.size(output) > SOURCE_SIZE, "Rewritten model did not grow.");
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes, "Rewritten FLOAT32 decoder");
      SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
      Operator gather = requireOperator(model, decode, 0, BuiltinOperator.GATHER);
      requireShape(decode.tensors(gather.inputs(1)), 1, TARGET_LENGTH);

      Operator positionAdd =
          requireOperator(model, decode, 1, BuiltinOperator.ADD);
      requireShape(
          decode.tensors(positionAdd.inputs(1)), 1, TARGET_LENGTH, HIDDEN_WIDTH);
      for (int index = 0; index < PROJECTION_RESHAPE_OPERATORS.length; index++) {
        Operator reshape =
            requireOperator(
                model,
                decode,
                PROJECTION_RESHAPE_OPERATORS[index],
                BuiltinOperator.RESHAPE);
        requireIntTensorData(
            model, decode.tensors(reshape.inputs(1)), 1, TARGET_LENGTH, HIDDEN_WIDTH);
        requireShape(
            decode.tensors(reshape.inputs(1)), 3);
        requireShape(
            decode.tensors(reshape.outputs(0)), 1, TARGET_LENGTH, HIDDEN_WIDTH);

        Operator projection =
            requireOperator(
                model,
                decode,
                PROJECTION_OPERATORS[index],
                BuiltinOperator.FULLY_CONNECTED);
        FullyConnectedOptions options =
            (FullyConnectedOptions)
                projection.builtinOptions(new FullyConnectedOptions());
        require(
            options != null && options.keepNumDims(),
            "Projection does not preserve the rewritten rank.");
        requireShape(
            decode.tensors(projection.outputs(0)),
            1,
            TARGET_LENGTH,
            HIDDEN_WIDTH);
      }
    }
  }

  private static Operator requireOperator(
      Model model, SubGraph graph, int index, int expectedBuiltin) {
    Operator operator = graph.operators(index);
    OperatorCode code = model.operatorCodes((int) operator.opcodeIndex());
    require(
        code.builtinCode() == expectedBuiltin,
        "Decoder operator " + index + " changed.");
    return operator;
  }

  private static void requireIntTensorData(
      Model model, Tensor tensor, int... expected) {
    require(tensor.type() == TensorType.INT32, "Shape tensor is not INT32.");
    Buffer buffer = model.buffers(Math.toIntExact(tensor.buffer()));
    require(
        buffer.dataLength() == expected.length * Integer.BYTES,
        "Shape tensor data size changed.");
    ByteBuffer data = buffer.dataAsByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
    for (int value : expected) {
      require(data.getInt() == value, "Shape tensor data changed.");
    }
  }

  private static ByteBuffer buildFullyConnectedOptions(
      FullyConnectedOptions source) {
    FlatBufferBuilder builder = new FlatBufferBuilder(64);
    int options =
        FullyConnectedOptions.createFullyConnectedOptions(
            builder,
            source.fusedActivationFunction(),
            source.weightsFormat(),
            true,
            source.asymmetricQuantizeInputs());
    builder.finish(options);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int appendIntVector(
      FileChannel channel, long offset, int[] values) throws IOException {
    require((offset & 3L) == 0L, "INT32 vector is not aligned.");
    ByteBuffer bytes =
        ByteBuffer.allocate(Integer.BYTES * (values.length + 1))
            .order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(values.length);
    for (int value : values) bytes.putInt(value);
    bytes.flip();
    writeFully(channel, bytes, offset);
    return Math.toIntExact(offset);
  }

  private static int appendByteVector(
      FileChannel channel, long offset, ByteBuffer values) throws IOException {
    require((offset & 3L) == 0L, "Byte vector is not aligned.");
    ByteBuffer bytes =
        ByteBuffer.allocate(Integer.BYTES + values.remaining())
            .order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(values.remaining());
    bytes.put(values.duplicate());
    bytes.flip();
    writeFully(channel, bytes, offset);
    return Math.toIntExact(offset);
  }

  private static ByteBuffer intBytes(int... values) {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Integer.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    for (int value : values) bytes.putInt(value);
    return bytes.flip();
  }

  private static void writeForwardOffset(
      FileChannel channel, int field, int target, String label)
      throws IOException {
    require(target > field, label + " target must point forward.");
    writeInt(channel, field, target - field);
  }

  private static void writeInt(FileChannel channel, long offset, int value)
      throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeFully(
      FileChannel channel, ByteBuffer source, long offset) throws IOException {
    long position = offset;
    while (source.hasRemaining()) {
      int written = channel.write(source, position);
      require(written > 0, "Could not write output model at byte " + position + ".");
      position += written;
    }
  }

  private static long align(long value, int alignment) {
    return Math.addExact(value, alignment - 1L) & -alignment;
  }

  private static boolean containsTensor(
      SubGraph graph, boolean inputs, int expectedTensor) {
    int length = inputs ? graph.inputsLength() : graph.outputsLength();
    for (int index = 0; index < length; index++) {
      int actual = inputs ? graph.inputs(index) : graph.outputs(index);
      if (actual == expectedTensor) return true;
    }
    return false;
  }

  private static void requireShape(Tensor tensor, int... expected) {
    require(
        tensor.shapeLength() == expected.length,
        "Tensor " + tensor.name() + " rank changed.");
    for (int index = 0; index < expected.length; index++) {
      require(
          tensor.shape(index) == expected[index],
          "Tensor " + tensor.name() + " shape changed.");
    }
  }

  private static Model parseModel(ByteBuffer bytes, String label) {
    require(
        Model.ModelBufferHasIdentifier(bytes.duplicate()),
        label + " does not have the TFL3 identifier.");
    Model model = Model.getRootAsModel(bytes);
    require(model.version() == 3, label + " schema version is not 3.");
    return model;
  }

  private static int tableFieldPosition(
      Table table, int vtableOffset, String label) {
    try {
      int relative = (int) TABLE_OFFSET.invoke(table, vtableOffset);
      require(relative != 0, label + " field is absent.");
      return Math.addExact(TABLE_POSITION.getInt(table), relative);
    } catch (ReflectiveOperationException error) {
      throw new ToolFailure("Could not inspect " + label + ".", error);
    }
  }

  private static boolean hasBuildPathSegment(Path path) {
    for (Path part : path) {
      if ("build".equals(part.toString())) return true;
    }
    return false;
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

  private record RewritePlan(
      List<Integer> rankedTensorShapeFields,
      int shapeTensorShapeField,
      int shapeTensorDataField,
      List<Integer> fullyConnectedOptionsFields,
      FullyConnectedOptions fullyConnectedOptions) {}

  private static final class ToolFailure extends RuntimeException {
    ToolFailure(String message) {
      super(message);
    }

    ToolFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
