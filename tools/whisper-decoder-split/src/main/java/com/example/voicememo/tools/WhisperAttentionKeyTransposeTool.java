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
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.tensorflow.lite.schema.AddOptions;
import org.tensorflow.lite.schema.BatchMatMulOptions;
import org.tensorflow.lite.schema.BuiltinOperator;
import org.tensorflow.lite.schema.BuiltinOptions;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.Conv2DOptions;
import org.tensorflow.lite.schema.CustomQuantization;
import org.tensorflow.lite.schema.FullyConnectedOptions;
import org.tensorflow.lite.schema.GatherOptions;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.MulOptions;
import org.tensorflow.lite.schema.Operator;
import org.tensorflow.lite.schema.OperatorCode;
import org.tensorflow.lite.schema.PadOptions;
import org.tensorflow.lite.schema.QuantizationDetails;
import org.tensorflow.lite.schema.QuantizationParameters;
import org.tensorflow.lite.schema.ReducerOptions;
import org.tensorflow.lite.schema.ReshapeOptions;
import org.tensorflow.lite.schema.SignatureDef;
import org.tensorflow.lite.schema.SoftmaxOptions;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.SubOptions;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorMap;
import org.tensorflow.lite.schema.TensorType;

/**
 * Rewrites Whisper attention scores from implicit {@code adj_y=true} to an
 * explicit key transpose followed by {@code adj_y=false}.
 *
 * <p>The inspect command is intentionally available separately so a changed
 * upstream model fails closed before any output is produced.
 */
public final class WhisperAttentionKeyTransposeTool {
  private static final List<SourceContract> PINNED_SOURCES =
      List.of(
          new SourceContract(
              "aligned FLOAT32 decoder diagnostic",
              964_989_184L,
              "b7d528f6b978bed123b30a24ed68a8c139eb9eabd75e57af43af78a82c7baeb0",
              new int[] {4}),
          new SourceContract(
              "production INT8 Whisper model",
              1_088_340_944L,
              "c40690d926fe9841ef6e0296eb0c4168c2910171c1d67b4bcf00a4b7cd7989ca",
              new int[] {8, 32}));
  private static final int GELU_OPTIONS_TYPE = 116;
  private static final int BLOCKWISE_QUANTIZATION_DETAILS_TYPE = 2;
  private static final int MULTI_AXIS_QUANTIZATION_DETAILS_TYPE = 3;
  private static final int[] KEY_PERMUTATION = {0, 2, 1};
  private static final int CURRENT_TENSOR_FIELD_COUNT = 11;
  private static final long REWRITTEN_PRODUCTION_SIZE = 1_094_571_944L;
  private static final String REWRITTEN_PRODUCTION_SHA256 =
      "8e69e43d7773b856a93cc192f6308adc1d1e6c921d48c91c684bc78bf6e91418";
  private static final long FLOAT_SELF_ATTENTION_SOURCE_SIZE = 1_120_786_928L;
  private static final String FLOAT_SELF_ATTENTION_SOURCE_SHA256 =
      "02ba9c2243d0bd4fce1c519fb12c3bfd1d105b4142e0a6b365537cde4c5a9e0f";
  private static final int REWRITTEN_DECODER_TENSOR_COUNT = 445;
  private static final int REWRITTEN_DECODER_OPERATOR_COUNT = 329;
  private static final int REWRITTEN_ENCODER_OPERATOR_COUNT = 1494;
  private static final int REWRITTEN_LOGITS_TENSOR = 434;
  private static final int FULL_DECODER_FLOAT_PROJECTION_COUNT = 41;
  private static final int FULL_DECODER_MASK_RESHAPE_COUNT = 4;
  private static final int FULL_DECODER_REWRITTEN_TENSOR_COUNT =
      REWRITTEN_DECODER_TENSOR_COUNT + FULL_DECODER_MASK_RESHAPE_COUNT;
  private static final int FULL_DECODER_REWRITTEN_OPERATOR_COUNT =
      REWRITTEN_DECODER_OPERATOR_COUNT + FULL_DECODER_MASK_RESHAPE_COUNT;
  private static final int FIRST_BLOCK_K_OPERATOR = 20;
  private static final int FIRST_BLOCK_K_INPUT_TENSOR = 133;
  private static final int FIRST_BLOCK_K_WEIGHT_TENSOR = 73;
  private static final int FIRST_BLOCK_K_OUTPUT_TENSOR = 134;
  private static final int FIRST_BLOCK_K_WEIGHT_BUFFER = 71;
  private static final int FIRST_BLOCK_K_CHANNELS = 1280;
  private static final int FIRST_BLOCK_K_WEIGHT_ELEMENTS =
      FIRST_BLOCK_K_CHANNELS * FIRST_BLOCK_K_CHANNELS;
  private static final long FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES =
      (long) FIRST_BLOCK_K_WEIGHT_ELEMENTS * Float.BYTES;
  private static final long K_PROJECTION_PROBE_SIZE = 1_654_640L;
  private static final String K_PROJECTION_PROBE_SHA256 =
      "7a30331ccb89e332edb176587d149f8d852e5b224864247defadae37db940b1d";
  private static final FloatProjectionSpec[] FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS = {
      new FloatProjectionSpec(
          "k",
          FIRST_BLOCK_K_OPERATOR,
          FIRST_BLOCK_K_INPUT_TENSOR,
          FIRST_BLOCK_K_WEIGHT_TENSOR,
          -1,
          FIRST_BLOCK_K_OUTPUT_TENSOR,
          FIRST_BLOCK_K_WEIGHT_BUFFER),
      new FloatProjectionSpec("q", 24, 133, 26, 25, 138, 24),
      new FloatProjectionSpec("v", 33, 133, 72, 41, 146, 70),
      new FloatProjectionSpec("out", 41, 153, 71, 24, 154, 69),
  };
  private static final int[] DECODER_HIDDEN_SHAPE = {1, 128, 1280};
  private static final DecoderPrefixStage[] DECODER_PREFIX_STAGES = {
      new DecoderPrefixStage("after-position-add", 2, 115, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-self-layer-norm", 20, 133, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-self-key-projection", 21, 134, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-self-query-projection", 25, 138, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage(
          "after-self-query-reshape", 28, 141, new int[] {20, 128, 64}),
      new DecoderPrefixStage(
          "after-self-key-transpose", 29, 437, new int[] {20, 64, 128}),
      new DecoderPrefixStage(
          "after-self-score-matmul", 30, 142, new int[] {20, 128, 128}),
      new DecoderPrefixStage(
          "after-self-mask-add", 31, 143, new int[] {1, 20, 128, 128}),
      new DecoderPrefixStage(
          "after-self-softmax", 33, 145, new int[] {20, 128, 128}),
      new DecoderPrefixStage("after-self-value-projection", 34, 146, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage(
          "after-self-value-reshape", 37, 149, new int[] {20, 128, 64}),
      new DecoderPrefixStage(
          "after-self-value-matmul", 38, 150, new int[] {20, 128, 64}),
      new DecoderPrefixStage(
          "after-self-context-reshape", 41, 153, new int[] {128, 1280}),
      new DecoderPrefixStage(
          "after-self-output-projection", 42, 154, new int[] {128, 1280}),
      new DecoderPrefixStage("after-self-attention", 43, 155, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-cross-attention", 67, 178, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-1-block", 81, 192, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-2-blocks", 160, 269, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-3-blocks", 239, 346, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-4-blocks", 318, 423, DECODER_HIDDEN_SHAPE),
      new DecoderPrefixStage("after-final-layer-norm", 328, 433, DECODER_HIDDEN_SHAPE),
  };

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

  private WhisperAttentionKeyTransposeTool() {}

  public static void main(String[] args) {
    try {
      if (args.length == 2 && "inspect".equals(args[0])) {
        inspect(Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 2 && "inspect-boundaries".equals(args[0])) {
        inspectDecoderBoundaries(
            Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 4 && "inspect-operators".equals(args[0])) {
        inspectDecoderOperators(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Integer.parseInt(args[2]),
            Integer.parseInt(args[3]));
      } else if (args.length == 3 && "inspect-tensor".equals(args[0])) {
        inspectDecoderTensor(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Integer.parseInt(args[2]));
      } else if (args.length == 4 && "generate-prefix".equals(args[0])) {
        generateDecoderPrefix(
            Path.of(args[1]).toAbsolutePath().normalize(),
            requireDecoderPrefixStage(args[2]),
            Path.of(args[3]).toAbsolutePath().normalize());
      } else if (args.length == 4 && "generate-float-k-prefix".equals(args[0])) {
        generateFloatKDecoderPrefix(
            Path.of(args[1]).toAbsolutePath().normalize(),
            requireDecoderPrefixStage(args[2]),
            Path.of(args[3]).toAbsolutePath().normalize());
      } else if (
          args.length == 4
              && "generate-float-self-attention-prefix".equals(args[0])) {
        generateFloatSelfAttentionDecoderPrefix(
            Path.of(args[1]).toAbsolutePath().normalize(),
            requireDecoderPrefixStage(args[2]),
            Path.of(args[3]).toAbsolutePath().normalize());
      } else if (
          args.length == 3
              && "generate-mask-add-rank-reshape-prefix".equals(args[0])) {
        generateMaskAddRankReshapeDecoderPrefix(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Path.of(args[2]).toAbsolutePath().normalize());
      } else if (
          args.length == 3
              && "generate-full-float-decoder".equals(args[0])) {
        generateFullFloatDecoder(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Path.of(args[2]).toAbsolutePath().normalize());
      } else if (args.length == 2 && "generate-softmax-probe".equals(args[0])) {
        generateSoftmaxProbe(
            Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 2 && "generate-score-probe".equals(args[0])) {
        generateScoreProbe(
            Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 2 && "generate-value-probe".equals(args[0])) {
        generateValueProbe(
            Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 2 && "generate-mask-add-probe".equals(args[0])) {
        generateMaskAddProbe(
            Path.of(args[1]).toAbsolutePath().normalize());
      } else if (args.length == 3 && "generate-k-projection-probe".equals(args[0])) {
        generateKProjectionProbe(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Path.of(args[2]).toAbsolutePath().normalize());
      } else if (
          args.length == 3
              && "generate-k-projection-arithmetic-probes".equals(args[0])) {
        generateKProjectionArithmeticProbes(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Path.of(args[2]).toAbsolutePath().normalize());
      } else if (args.length == 3 && "generate".equals(args[0])) {
        generate(
            Path.of(args[1]).toAbsolutePath().normalize(),
            Path.of(args[2]).toAbsolutePath().normalize());
      } else if (
          args.length == 5 && "generate-verified".equals(args[0])) {
        generateVerified(
            Path.of(args[1]).toAbsolutePath().normalize(),
            args[2],
            parseRewriteCounts(args[3]),
            Path.of(args[4]).toAbsolutePath().normalize());
      } else if (
          args.length == 4
              && "generate-mask-rank-reshapes-verified".equals(args[0])) {
        generateMaskRankReshapesVerified(
            Path.of(args[1]).toAbsolutePath().normalize(),
            args[2],
            Path.of(args[3]).toAbsolutePath().normalize());
      } else {
        throw new ToolFailure(
            "Usage: WhisperAttentionKeyTransposeTool inspect MODEL\n"
                + "   or: WhisperAttentionKeyTransposeTool inspect-boundaries MODEL\n"
                + "   or: WhisperAttentionKeyTransposeTool inspect-operators "
                + "MODEL START END_EXCLUSIVE\n"
                + "   or: WhisperAttentionKeyTransposeTool inspect-tensor "
                + "MODEL TENSOR_INDEX\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-prefix "
                + "SOURCE STAGE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-float-k-prefix "
                + "SOURCE STAGE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool "
                + "generate-float-self-attention-prefix SOURCE STAGE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool "
                + "generate-mask-add-rank-reshape-prefix SOURCE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool "
                + "generate-full-float-decoder SOURCE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-softmax-probe "
                + "OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-score-probe "
                + "OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-value-probe "
                + "OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-mask-add-probe "
                + "OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-k-projection-probe "
                + "SOURCE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool "
                + "generate-k-projection-arithmetic-probes PROBE OUTPUT_DIRECTORY\n"
                + "   or: WhisperAttentionKeyTransposeTool generate SOURCE OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool generate-verified "
                + "SOURCE EXPECTED_SHA256 REWRITES_PER_SUBGRAPH OUTPUT\n"
                + "   or: WhisperAttentionKeyTransposeTool "
                + "generate-mask-rank-reshapes-verified "
                + "SOURCE EXPECTED_SHA256 OUTPUT");
      }
    } catch (ToolFailure error) {
      System.err.println("Whisper attention rewrite failed: " + error.getMessage());
      System.exit(2);
    } catch (Exception error) {
      System.err.println(
          "Whisper attention rewrite failed: "
              + error.getClass().getSimpleName()
              + ": "
              + error.getMessage());
      System.exit(3);
    }
  }

  private static void generate(Path source, Path output) throws Exception {
    require(Files.isRegularFile(source), "Source model is missing: " + source);
    require(!source.equals(output), "Output must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "Rewritten output must stay under a build directory.");
    SourceContract contract = requirePinnedSource(source);
    generate(source, output, contract);
  }

  private static void generateVerified(
      Path source,
      String expectedSha256,
      int[] rewritesPerSubgraph,
      Path output)
      throws Exception {
    require(Files.isRegularFile(source), "Source model is missing: " + source);
    requireSha256(expectedSha256);
    String actualSha256 = sha256(source);
    require(
        expectedSha256.equals(actualSha256),
        "Source SHA-256 changed: expected "
            + expectedSha256
            + ", found "
            + actualSha256
            + ".");
    require(
        rewritesPerSubgraph.length > 0,
        "At least one per-subgraph rewrite count is required.");
    for (int count : rewritesPerSubgraph) {
      require(count >= 0, "Rewrite counts must not be negative.");
    }
    generate(
        source,
        output,
        new SourceContract(
            "hash-verified FP16 experiment",
            Files.size(source),
            expectedSha256,
            rewritesPerSubgraph));
  }

  private static void generate(
      Path source, Path output, SourceContract contract) throws Exception {
    require(Files.isRegularFile(source), "Source model is missing: " + source);
    require(!source.equals(output), "Output must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "Rewritten output must stay under a build directory.");
    require(
        Files.size(source) == contract.size()
            && contract.sha256().equals(sha256(source)),
        "Source model changed after its contract was established.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        validateSourceContract(sourceModel, contract);
        int subgraphVector = tableVectorStart(sourceModel, 8, "model subgraphs");
        for (int graphIndex = 0;
            graphIndex < sourceModel.subgraphsLength();
            graphIndex++) {
          SubGraph graph = sourceModel.subgraphs(graphIndex);
          ByteBuffer fragment = buildRewrittenSubgraph(sourceModel, graph);
          long appendOffset = align(outputChannel.size(), Integer.BYTES);
          if (appendOffset > outputChannel.size()) {
            writeFully(
                outputChannel,
                ByteBuffer.allocate(
                    Math.toIntExact(appendOffset - outputChannel.size())),
                outputChannel.size());
          }
          int fragmentRoot = fragment.getInt(fragment.position());
          require(
              fragmentRoot > 0 && fragmentRoot < fragment.remaining(),
              "Generated subgraph fragment root is invalid.");
          writeFully(outputChannel, fragment, appendOffset);
          int vectorElement =
              Math.addExact(
                  subgraphVector + Integer.BYTES,
                  Math.multiplyExact(graphIndex, Integer.BYTES));
          int target = Math.toIntExact(appendOffset + fragmentRoot);
          writeForwardOffset(
              outputChannel,
              vectorElement,
              target,
              "rewritten subgraph " + graphIndex);
        }
        outputChannel.force(true);
      }
      verifyOutput(source, temporary, contract);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("source_contract=" + contract.label());
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("explicit_key_transposes=" + contract.totalRewrites());
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static int[] parseRewriteCounts(String value) {
    require(
        value != null && value.matches("[0-9]+(?:,[0-9]+)*"),
        "Rewrite counts must be comma-separated non-negative integers.");
    String[] parts = value.split(",");
    int[] counts = new int[parts.length];
    try {
      for (int index = 0; index < parts.length; index++) {
        counts[index] = Integer.parseInt(parts[index]);
      }
    } catch (NumberFormatException error) {
      throw new ToolFailure("Rewrite count is outside the supported range.", error);
    }
    return counts;
  }

  private static void requireSha256(String value) {
    require(
        value != null && value.matches("[0-9a-f]{64}"),
        "Expected SHA-256 must be 64 lowercase hexadecimal characters.");
  }

  private static SourceContract requirePinnedSource(Path source)
      throws IOException {
    long size = Files.size(source);
    String hash = sha256(source);
    for (SourceContract contract : PINNED_SOURCES) {
      if (contract.size() == size && contract.sha256().equals(hash)) {
        return contract;
      }
    }
    throw new ToolFailure(
        "Source model is not a pinned VoiceMemo artifact: "
            + size
            + " bytes, SHA-256 "
            + hash
            + ".");
  }

  private static void validateSourceContract(
      Model model, SourceContract contract) {
    require(
        model.subgraphsLength() == contract.rewritesPerSubgraph().length,
        contract.label() + " subgraph count changed.");
    require(findOperatorCode(model, BuiltinOperator.TRANSPOSE) >= 0,
        "Source model has no TRANSPOSE operator code.");
    require(findPermutationBuffer(model) >= 0,
        "Source model has no reusable INT32 [0,2,1] buffer.");
    for (int graphIndex = 0;
        graphIndex < model.subgraphsLength();
        graphIndex++) {
      SubGraph graph = model.subgraphs(graphIndex);
      validateCloneableGraph(graph);
      List<Integer> targets = findRewriteTargets(model, graph);
      require(
          targets.size() == contract.rewritesPerSubgraph()[graphIndex],
          "Subgraph "
              + graphIndex
              + " adj_y=true count changed: "
              + targets.size()
              + ".");
    }
  }

  private static void validateCloneableGraph(SubGraph graph) {
    require(
        !fieldPresent(graph, 14) || readIntField(graph, 14, -1) == -1,
        "Subgraph debug metadata is not supported by the metadata-only rewrite.");
    for (int index = 0; index < graph.tensorsLength(); index++) {
      Tensor tensor = graph.tensors(index);
      require(
          !fieldPresent(tensor, 22),
          "Tensor variant metadata is not supported by the rewrite.");
      require(
          !fieldPresent(tensor, 24)
              || Integer.toUnsignedLong(readIntField(tensor, 24, 0)) == 0,
          "External tensor buffers are not supported by the rewrite.");
      require(
          tensor.sparsity() == null,
          "Sparse tensor metadata is not supported by the rewrite.");
      QuantizationParameters quantization = tensor.quantization();
      require(
          quantization == null
              || quantization.detailsType() == QuantizationDetails.NONE
              || quantization.detailsType()
                  == QuantizationDetails.CustomQuantization
              || quantization.detailsType()
                  == BLOCKWISE_QUANTIZATION_DETAILS_TYPE
              || quantization.detailsType()
                  == MULTI_AXIS_QUANTIZATION_DETAILS_TYPE,
          "Unsupported tensor quantization details type.");
    }
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator operator = graph.operators(index);
      require(
          !fieldPresent(operator, 22)
              || readLongField(operator, 22, 0L) == 0L,
          "Large custom operator options are not supported.");
      require(
          !fieldPresent(operator, 24)
              || readLongField(operator, 24, 0L) == 0L,
          "Large custom operator option sizes are not supported.");
      require(
          !fieldPresent(operator, 26)
              || readByteField(operator, 26, (byte) 0) == 0,
          "BuiltinOptions2 operators are not supported.");
      require(
          !fieldPresent(operator, 28),
          "BuiltinOptions2 payloads are not supported.");
      require(
          !fieldPresent(operator, 30)
              || readIntField(operator, 30, -1) == -1,
          "Operator debug metadata is not supported.");
      requireSupportedOptions(operator.builtinOptionsType());
    }
  }

  private static void requireSupportedOptions(byte optionType) {
    if (optionType == BuiltinOptions.NONE
        || optionType == BuiltinOptions.AddOptions
        || optionType == BuiltinOptions.BatchMatMulOptions
        || optionType == BuiltinOptions.Conv2DOptions
        || optionType == BuiltinOptions.FullyConnectedOptions
        || optionType == BuiltinOptions.GatherOptions
        || optionType == BuiltinOptions.MulOptions
        || optionType == BuiltinOptions.PadOptions
        || optionType == BuiltinOptions.ReducerOptions
        || optionType == BuiltinOptions.ReshapeOptions
        || optionType == BuiltinOptions.SoftmaxOptions
        || optionType == BuiltinOptions.SubOptions
        || Byte.toUnsignedInt(optionType) == GELU_OPTIONS_TYPE) {
      return;
    }
    throw new ToolFailure(
        "Unsupported builtin options type "
            + Byte.toUnsignedInt(optionType)
            + ".");
  }

  private static List<Integer> findRewriteTargets(
      Model model, SubGraph graph) {
    List<Integer> targets = new ArrayList<>();
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator operator = graph.operators(index);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(operator.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.BATCH_MATMUL) continue;
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(options != null, "BATCH_MATMUL " + index + " has no options.");
      if (!options.adjY()) continue;
      require(!options.adjX(), "Whisper score BATCH_MATMUL has adj_x=true.");
      require(operator.inputsLength() == 2, "BATCH_MATMUL input count changed.");
      require(operator.outputsLength() == 1, "BATCH_MATMUL output count changed.");
      Tensor key = graph.tensors(operator.inputs(1));
      require(key.type() == TensorType.FLOAT32,
          "Whisper attention key is not FLOAT32.");
      require(key.shapeLength() == 3,
          "Whisper attention key is not rank 3.");
      targets.add(index);
    }
    return targets;
  }

  private static ByteBuffer buildRewrittenSubgraph(
      Model model, SubGraph graph) {
    List<Integer> targets = findRewriteTargets(model, graph);
    require(!targets.isEmpty(), "Subgraph has no adj_y=true score operations.");
    int permutationTensor = findPermutationTensor(model, graph);
    boolean addPermutationTensor = permutationTensor < 0;
    if (addPermutationTensor) {
      permutationTensor = graph.tensorsLength();
    }
    int firstKeyOutput =
        graph.tensorsLength() + (addPermutationTensor ? 1 : 0);
    Map<Integer, Integer> outputForOperator = new LinkedHashMap<>();
    for (int index = 0; index < targets.size(); index++) {
      outputForOperator.put(targets.get(index), firstKeyOutput + index);
    }

    FlatBufferBuilder builder = new FlatBufferBuilder(16 * 1024 * 1024);
    int[] tensorTables =
        new int[
            graph.tensorsLength()
                + (addPermutationTensor ? 1 : 0)
                + targets.size()];
    for (int index = 0; index < graph.tensorsLength(); index++) {
      tensorTables[index] = cloneTensor(builder, graph.tensors(index));
    }
    if (addPermutationTensor) {
      tensorTables[permutationTensor] =
          createTensor(
              builder,
              KEY_PERMUTATION.length == 3 ? new int[] {3} : new int[0],
              TensorType.INT32,
              findPermutationBuffer(model),
              "voicememo_attention_key_permutation",
              0,
              false,
              null,
              true,
              true);
    }
    for (int target : targets) {
      Operator score = graph.operators(target);
      Tensor key = graph.tensors(score.inputs(1));
      int[] transposedShape = tensorShape(key);
      int last = transposedShape.length - 1;
      int swap = transposedShape[last];
      transposedShape[last] = transposedShape[last - 1];
      transposedShape[last - 1] = swap;
      int outputTensor = outputForOperator.get(target);
      tensorTables[outputTensor] =
          createTensor(
              builder,
              transposedShape,
              key.type(),
              0,
              key.name() + "/voicememo_explicit_transpose",
              0,
              false,
              null,
              true,
              true);
    }
    int tensors = SubGraph.createTensorsVector(builder, tensorTables);

    int transposeOpcode = findOperatorCode(model, BuiltinOperator.TRANSPOSE);
    int[] operatorTables =
        new int[graph.operatorsLength() + targets.size()];
    int destination = 0;
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator source = graph.operators(index);
      Integer transposedKey = outputForOperator.get(index);
      if (transposedKey != null) {
        operatorTables[destination++] =
            createOperator(
                builder,
                transposeOpcode,
                new int[] {source.inputs(1), permutationTensor},
                new int[] {transposedKey},
                BuiltinOptions.NONE,
                0,
                new byte[0],
                (byte) 0,
                new boolean[0],
                new int[0]);
      }
      operatorTables[destination++] =
          cloneOperator(builder, source, transposedKey);
    }
    require(destination == operatorTables.length,
        "Rewritten operator count is inconsistent.");
    int operators = SubGraph.createOperatorsVector(builder, operatorTables);
    int inputs =
        SubGraph.createInputsVector(builder, operatorInputs(graph, true));
    int outputs =
        SubGraph.createOutputsVector(builder, operatorInputs(graph, false));
    int name = graph.name() == null ? 0 : builder.createString(graph.name());
    int subgraph =
        SubGraph.createSubGraph(
            builder, tensors, inputs, outputs, operators, name);
    builder.finish(subgraph);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static int cloneTensor(FlatBufferBuilder builder, Tensor source) {
    return cloneTensorWithBuffer(
        builder, source, Math.toIntExact(source.buffer()));
  }

  private static int cloneTensorWithBuffer(
      FlatBufferBuilder builder, Tensor source, int buffer) {
    int quantization = cloneQuantization(builder, source.quantization());
    int[] shapeSignature =
        source.shapeSignatureLength() == 0
            ? null
            : tensorShapeSignature(source);
    boolean hasRankPresent = fieldPresent(source, 20);
    boolean hasRank = readBooleanField(source, 20, false);
    return createTensor(
        builder,
        tensorShape(source),
        source.type(),
        buffer,
        source.name(),
        quantization,
        source.isVariable(),
        shapeSignature,
        hasRank,
        hasRankPresent);
  }

  private static int cloneTensorWithShape(
      FlatBufferBuilder builder, Tensor source, int[] shape) {
    int quantization = cloneQuantization(builder, source.quantization());
    boolean hasRankPresent = fieldPresent(source, 20);
    boolean hasRank = readBooleanField(source, 20, false);
    return createTensor(
        builder,
        shape,
        source.type(),
        Math.toIntExact(source.buffer()),
        source.name() + "/voicememo_explicit_rank4",
        quantization,
        source.isVariable(),
        source.shapeSignatureLength() == 0 ? null : shape,
        hasRank,
        hasRankPresent);
  }

  private static int cloneTensorAsFloat(
      FlatBufferBuilder builder, Tensor source) {
    require(
        source.type() == TensorType.INT8 && source.shapeLength() == 2,
        "Only rank-2 INT8 decoder weights can be converted to FLOAT32.");
    boolean hasRankPresent = fieldPresent(source, 20);
    boolean hasRank = readBooleanField(source, 20, false);
    return createTensor(
        builder,
        tensorShape(source),
        TensorType.FLOAT32,
        Math.toIntExact(source.buffer()),
        source.name() + "/voicememo_dequantized_float32",
        0,
        source.isVariable(),
        source.shapeSignatureLength() == 0
            ? null
            : tensorShapeSignature(source),
        hasRank,
        hasRankPresent);
  }

  private static int createTensor(
      FlatBufferBuilder builder,
      int[] shape,
      byte type,
      int buffer,
      String nameValue,
      int quantization,
      boolean isVariable,
      int[] shapeSignature,
      boolean hasRank,
      boolean hasRankPresent) {
    int name = nameValue == null ? 0 : builder.createString(nameValue);
    int shapeOffset = Tensor.createShapeVector(builder, shape);
    int shapeSignatureOffset =
        shapeSignature == null
            ? 0
            : Tensor.createShapeSignatureVector(builder, shapeSignature);
    builder.startTable(CURRENT_TENSOR_FIELD_COUNT);
    if (hasRankPresent) builder.addBoolean(8, hasRank, false);
    if (shapeSignatureOffset != 0) builder.addOffset(7, shapeSignatureOffset, 0);
    if (isVariable) builder.addBoolean(5, true, false);
    if (quantization != 0) builder.addOffset(4, quantization, 0);
    if (name != 0) builder.addOffset(3, name, 0);
    if (buffer != 0) builder.addInt(2, buffer, 0);
    builder.addByte(1, type, 0);
    builder.addOffset(0, shapeOffset, 0);
    return builder.endTable();
  }

  private static int cloneQuantization(
      FlatBufferBuilder builder, QuantizationParameters source) {
    if (source == null) return 0;
    int minimum =
        source.minLength() == 0
            ? 0
            : QuantizationParameters.createMinVector(
                builder, floatVector(source, 0));
    int maximum =
        source.maxLength() == 0
            ? 0
            : QuantizationParameters.createMaxVector(
                builder, floatVector(source, 1));
    float[] scales = new float[source.scaleLength()];
    for (int index = 0; index < scales.length; index++) {
      scales[index] = source.scale(index);
    }
    int scale =
        scales.length == 0
            ? 0
            : QuantizationParameters.createScaleVector(builder, scales);
    long[] zeroPoints = new long[source.zeroPointLength()];
    for (int index = 0; index < zeroPoints.length; index++) {
      zeroPoints[index] = source.zeroPoint(index);
    }
    int zeroPoint =
        zeroPoints.length == 0
            ? 0
            : QuantizationParameters.createZeroPointVector(builder, zeroPoints);
    int details = 0;
    if (source.detailsType() == QuantizationDetails.CustomQuantization) {
      CustomQuantization custom =
          (CustomQuantization)
              source.details(new CustomQuantization());
      require(custom != null, "Custom quantization payload is missing.");
      byte[] values = new byte[custom.customLength()];
      for (int index = 0; index < values.length; index++) {
        values[index] = (byte) custom.custom(index);
      }
      int data = CustomQuantization.createCustomVector(builder, values);
      details = CustomQuantization.createCustomQuantization(builder, data);
    } else if (source.detailsType()
        == BLOCKWISE_QUANTIZATION_DETAILS_TYPE) {
      Table blockwise = source.details(new Table());
      require(blockwise != null, "Blockwise quantization payload is missing.");
      builder.startTable(3);
      builder.addInt(2, readIntField(blockwise, 8, 0), 0);
      builder.addInt(1, readIntField(blockwise, 6, 0), 0);
      builder.addInt(0, readIntField(blockwise, 4, 0), 0);
      details = builder.endTable();
    } else if (source.detailsType()
        == MULTI_AXIS_QUANTIZATION_DETAILS_TYPE) {
      Table multiAxis = source.details(new Table());
      require(multiAxis != null, "Multi-axis quantization payload is missing.");
      int[] dimensions = readIntVector(multiAxis, 10);
      int dimensionsOffset =
          dimensions.length == 0
              ? 0
              : createIntVector(builder, dimensions);
      builder.startTable(4);
      if (dimensionsOffset != 0) builder.addOffset(3, dimensionsOffset, 0);
      builder.addInt(2, readIntField(multiAxis, 8, 0), 0);
      builder.addInt(1, readIntField(multiAxis, 6, 0), 0);
      builder.addInt(0, readIntField(multiAxis, 4, 0), 0);
      details = builder.endTable();
    }
    return QuantizationParameters.createQuantizationParameters(
        builder,
        minimum,
        maximum,
        scale,
        zeroPoint,
        source.detailsType(),
        details,
        source.quantizedDimension());
  }

  private static float[] floatVector(
      QuantizationParameters source, int kind) {
    int length = kind == 0 ? source.minLength() : source.maxLength();
    float[] values = new float[length];
    for (int index = 0; index < values.length; index++) {
      values[index] = kind == 0 ? source.min(index) : source.max(index);
    }
    return values;
  }

  private static int cloneOperator(
      FlatBufferBuilder builder,
      Operator source,
      Integer transposedKey) {
    int[] inputs = operatorVector(source, true);
    if (transposedKey != null) {
      require(inputs.length == 2, "Rewritten BATCH_MATMUL input count changed.");
      inputs[1] = transposedKey;
    }
    int options =
        cloneBuiltinOptions(builder, source, transposedKey != null);
    byte[] customOptions = new byte[source.customOptionsLength()];
    for (int index = 0; index < customOptions.length; index++) {
      customOptions[index] = (byte) source.customOptions(index);
    }
    boolean[] mutating = new boolean[source.mutatingVariableInputsLength()];
    for (int index = 0; index < mutating.length; index++) {
      mutating[index] = source.mutatingVariableInputs(index);
    }
    return createOperator(
        builder,
        Math.toIntExact(source.opcodeIndex()),
        inputs,
        operatorVector(source, false),
        source.builtinOptionsType(),
        options,
        customOptions,
        source.customOptionsFormat(),
        mutating,
        intermediateVector(source));
  }

  private static int cloneOperatorReplacingInput(
      FlatBufferBuilder builder,
      Operator source,
      int inputPosition,
      int replacementTensor) {
    int[] inputs = operatorVector(source, true);
    require(
        inputPosition >= 0 && inputPosition < inputs.length,
        "Replacement operator input position is invalid.");
    inputs[inputPosition] = replacementTensor;
    int options = cloneBuiltinOptions(builder, source, false);
    byte[] customOptions = new byte[source.customOptionsLength()];
    for (int index = 0; index < customOptions.length; index++) {
      customOptions[index] = (byte) source.customOptions(index);
    }
    boolean[] mutating = new boolean[source.mutatingVariableInputsLength()];
    for (int index = 0; index < mutating.length; index++) {
      mutating[index] = source.mutatingVariableInputs(index);
    }
    return createOperator(
        builder,
        Math.toIntExact(source.opcodeIndex()),
        inputs,
        operatorVector(source, false),
        source.builtinOptionsType(),
        options,
        customOptions,
        source.customOptionsFormat(),
        mutating,
        intermediateVector(source));
  }

  private static int cloneBuiltinOptions(
      FlatBufferBuilder builder, Operator operator, boolean clearAdjY) {
    byte type = operator.builtinOptionsType();
    if (type == BuiltinOptions.NONE) return 0;
    if (type == BuiltinOptions.AddOptions) {
      AddOptions source =
          (AddOptions) operator.builtinOptions(new AddOptions());
      return AddOptions.createAddOptions(
          builder, source.fusedActivationFunction(), source.potScaleInt16());
    }
    if (type == BuiltinOptions.BatchMatMulOptions) {
      BatchMatMulOptions source =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      return BatchMatMulOptions.createBatchMatMulOptions(
          builder,
          source.adjX(),
          clearAdjY ? false : source.adjY(),
          source.asymmetricQuantizeInputs());
    }
    if (type == BuiltinOptions.Conv2DOptions) {
      Conv2DOptions source =
          (Conv2DOptions) operator.builtinOptions(new Conv2DOptions());
      return Conv2DOptions.createConv2DOptions(
          builder,
          source.padding(),
          source.strideW(),
          source.strideH(),
          source.fusedActivationFunction(),
          source.dilationWFactor(),
          source.dilationHFactor());
    }
    if (type == BuiltinOptions.FullyConnectedOptions) {
      FullyConnectedOptions source =
          (FullyConnectedOptions)
              operator.builtinOptions(new FullyConnectedOptions());
      return FullyConnectedOptions.createFullyConnectedOptions(
          builder,
          source.fusedActivationFunction(),
          source.weightsFormat(),
          source.keepNumDims(),
          source.asymmetricQuantizeInputs());
    }
    if (type == BuiltinOptions.GatherOptions) {
      GatherOptions source =
          (GatherOptions) operator.builtinOptions(new GatherOptions());
      return GatherOptions.createGatherOptions(
          builder, source.axis(), source.batchDims());
    }
    if (type == BuiltinOptions.MulOptions) {
      MulOptions source =
          (MulOptions) operator.builtinOptions(new MulOptions());
      return MulOptions.createMulOptions(
          builder, source.fusedActivationFunction());
    }
    if (type == BuiltinOptions.PadOptions) {
      PadOptions.startPadOptions(builder);
      return PadOptions.endPadOptions(builder);
    }
    if (type == BuiltinOptions.ReducerOptions) {
      ReducerOptions source =
          (ReducerOptions) operator.builtinOptions(new ReducerOptions());
      return ReducerOptions.createReducerOptions(builder, source.keepDims());
    }
    if (type == BuiltinOptions.ReshapeOptions) {
      ReshapeOptions source =
          (ReshapeOptions) operator.builtinOptions(new ReshapeOptions());
      int[] shape = new int[source.newShapeLength()];
      for (int index = 0; index < shape.length; index++) {
        shape[index] = source.newShape(index);
      }
      int newShape =
          shape.length == 0
              ? 0
              : ReshapeOptions.createNewShapeVector(builder, shape);
      return ReshapeOptions.createReshapeOptions(builder, newShape);
    }
    if (type == BuiltinOptions.SoftmaxOptions) {
      SoftmaxOptions source =
          (SoftmaxOptions) operator.builtinOptions(new SoftmaxOptions());
      return SoftmaxOptions.createSoftmaxOptions(builder, source.beta());
    }
    if (type == BuiltinOptions.SubOptions) {
      SubOptions source =
          (SubOptions) operator.builtinOptions(new SubOptions());
      return SubOptions.createSubOptions(
          builder, source.fusedActivationFunction(), source.potScaleInt16());
    }
    if (Byte.toUnsignedInt(type) == GELU_OPTIONS_TYPE) {
      Table source = operator.builtinOptions(new Table());
      require(source != null, "GELU options are missing.");
      boolean approximate = readBooleanField(source, 4, false);
      builder.startTable(1);
      builder.addBoolean(0, approximate, false);
      return builder.endTable();
    }
    throw new ToolFailure(
        "Unsupported builtin options type " + Byte.toUnsignedInt(type) + ".");
  }

  private static int createOperator(
      FlatBufferBuilder builder,
      int opcodeIndex,
      int[] inputs,
      int[] outputs,
      byte optionsType,
      int options,
      byte[] customOptions,
      byte customOptionsFormat,
      boolean[] mutating,
      int[] intermediates) {
    int inputVector = Operator.createInputsVector(builder, inputs);
    int outputVector = Operator.createOutputsVector(builder, outputs);
    int customVector =
        customOptions.length == 0
            ? 0
            : Operator.createCustomOptionsVector(builder, customOptions);
    int mutatingVector =
        mutating.length == 0
            ? 0
            : Operator.createMutatingVariableInputsVector(builder, mutating);
    int intermediateVector =
        intermediates.length == 0
            ? 0
            : Operator.createIntermediatesVector(builder, intermediates);
    return Operator.createOperator(
        builder,
        opcodeIndex,
        inputVector,
        outputVector,
        optionsType,
        options,
        customVector,
        customOptionsFormat,
        mutatingVector,
        intermediateVector);
  }

  private static int[] operatorInputs(SubGraph graph, boolean inputs) {
    int length = inputs ? graph.inputsLength() : graph.outputsLength();
    int[] values = new int[length];
    for (int index = 0; index < length; index++) {
      values[index] = inputs ? graph.inputs(index) : graph.outputs(index);
    }
    return values;
  }

  private static int[] operatorVector(Operator operator, boolean inputs) {
    int length = inputs ? operator.inputsLength() : operator.outputsLength();
    int[] values = new int[length];
    for (int index = 0; index < length; index++) {
      values[index] = inputs ? operator.inputs(index) : operator.outputs(index);
    }
    return values;
  }

  private static int[] intermediateVector(Operator operator) {
    int[] values = new int[operator.intermediatesLength()];
    for (int index = 0; index < values.length; index++) {
      values[index] = operator.intermediates(index);
    }
    return values;
  }

  private static int[] tensorShape(Tensor tensor) {
    int[] shape = new int[tensor.shapeLength()];
    for (int index = 0; index < shape.length; index++) {
      shape[index] = tensor.shape(index);
    }
    return shape;
  }

  private static int[] tensorShapeSignature(Tensor tensor) {
    int[] shape = new int[tensor.shapeSignatureLength()];
    for (int index = 0; index < shape.length; index++) {
      shape[index] = tensor.shapeSignature(index);
    }
    return shape;
  }

  private static int findOperatorCode(Model model, int builtinCode) {
    for (int index = 0; index < model.operatorCodesLength(); index++) {
      if (model.operatorCodes(index).builtinCode() == builtinCode) return index;
    }
    return -1;
  }

  private static void verifyOutput(
      Path source, Path output, SourceContract contract) throws Exception {
    require(Files.size(output) > Files.size(source),
        "Rewritten model did not grow.");
    try (FileChannel sourceChannel =
            FileChannel.open(source, StandardOpenOption.READ);
        FileChannel outputChannel =
            FileChannel.open(output, StandardOpenOption.READ)) {
      Model sourceModel =
          parseModel(
              sourceChannel
                  .map(
                      FileChannel.MapMode.READ_ONLY,
                      0,
                      sourceChannel.size())
                  .order(ByteOrder.LITTLE_ENDIAN));
      Model outputModel =
          parseModel(
              outputChannel
                  .map(
                      FileChannel.MapMode.READ_ONLY,
                      0,
                      outputChannel.size())
                  .order(ByteOrder.LITTLE_ENDIAN));
      require(
          outputModel.subgraphsLength() == sourceModel.subgraphsLength(),
          "Rewritten subgraph count changed.");
      for (int graphIndex = 0;
          graphIndex < outputModel.subgraphsLength();
          graphIndex++) {
        SubGraph before = sourceModel.subgraphs(graphIndex);
        SubGraph after = outputModel.subgraphs(graphIndex);
        int expected = contract.rewritesPerSubgraph()[graphIndex];
        require(
            after.tensorsLength() == before.tensorsLength() + expected
                + (findPermutationTensor(sourceModel, before) < 0 ? 1 : 0),
            "Rewritten tensor count changed unexpectedly.");
        require(
            after.operatorsLength() == before.operatorsLength() + expected,
            "Rewritten operator count changed unexpectedly.");
        require(
            Arrays.equals(operatorInputs(before, true), operatorInputs(after, true)),
            "Subgraph inputs changed.");
        require(
            Arrays.equals(
                operatorInputs(before, false), operatorInputs(after, false)),
            "Subgraph outputs changed.");
        verifyExplicitTransposes(outputModel, after, expected);
      }
    }
  }

  private static void verifyExplicitTransposes(
      Model model, SubGraph graph, int expected) {
    Map<Integer, Integer> producer = new LinkedHashMap<>();
    for (int operatorIndex = 0;
        operatorIndex < graph.operatorsLength();
        operatorIndex++) {
      Operator operator = graph.operators(operatorIndex);
      for (int outputIndex = 0;
          outputIndex < operator.outputsLength();
          outputIndex++) {
        producer.put(operator.outputs(outputIndex), operatorIndex);
      }
    }
    int explicitScores = 0;
    for (int operatorIndex = 0;
        operatorIndex < graph.operatorsLength();
        operatorIndex++) {
      Operator operator = graph.operators(operatorIndex);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(operator.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.BATCH_MATMUL) continue;
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(options != null && !options.adjY(),
          "Rewritten graph still contains adj_y=true.");
      Tensor right = graph.tensors(operator.inputs(1));
      if (right.name() == null
          || !right.name().endsWith("/voicememo_explicit_transpose")) {
        continue;
      }
      Integer transposeIndex = producer.get(operator.inputs(1));
      require(transposeIndex != null,
          "Rewritten score key has no producer.");
      Operator transpose = graph.operators(transposeIndex);
      require(
          model
                  .operatorCodes(Math.toIntExact(transpose.opcodeIndex()))
                  .builtinCode()
              == BuiltinOperator.TRANSPOSE,
          "Rewritten score key is not produced by TRANSPOSE.");
      require(transpose.inputsLength() == 2,
          "Explicit key TRANSPOSE input count changed.");
      Tensor permutation = graph.tensors(transpose.inputs(1));
      requirePermutationTensor(model, permutation);
      explicitScores++;
    }
    require(
        explicitScores == expected,
        "Expected "
            + expected
            + " explicit score transposes, found "
            + explicitScores
            + ".");
  }

  private static void requirePermutationTensor(Model model, Tensor tensor) {
    require(tensor.type() == TensorType.INT32,
        "Transpose permutation is not INT32.");
    require(Arrays.equals(tensorShape(tensor), new int[] {3}),
        "Transpose permutation shape changed.");
    Buffer buffer = model.buffers(Math.toIntExact(tensor.buffer()));
    require(buffer.dataLength() == 3 * Integer.BYTES,
        "Transpose permutation byte size changed.");
    ByteBuffer values =
        buffer.dataAsByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
    for (int expected : KEY_PERMUTATION) {
      require(values.getInt() == expected,
          "Transpose permutation data changed.");
    }
  }

  private static void inspect(Path path) throws Exception {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      System.out.println("bytes=" + channel.size());
      System.out.println("subgraphs=" + model.subgraphsLength());
      System.out.println("operator_codes=" + model.operatorCodesLength());
      for (int graphIndex = 0; graphIndex < model.subgraphsLength(); graphIndex++) {
        inspectGraph(model, model.subgraphs(graphIndex), graphIndex);
      }
    }
  }

  private static void inspectDecoderBoundaries(Path path) throws Exception {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() >= 1, "Decoder subgraph is missing.");
      SubGraph decode = model.subgraphs(0);
      Map<Integer, Integer> producerByTensor = new LinkedHashMap<>();
      for (int operatorIndex = 0;
          operatorIndex < decode.operatorsLength();
          operatorIndex++) {
        Operator operator = decode.operators(operatorIndex);
        for (int outputIndex = 0;
            outputIndex < operator.outputsLength();
            outputIndex++) {
          producerByTensor.put(operator.outputs(outputIndex), operatorIndex);
        }
      }
      System.out.println("bytes=" + channel.size());
      System.out.println("decode_tensors=" + decode.tensorsLength());
      System.out.println("decode_operators=" + decode.operatorsLength());
      for (int outputIndex = 0;
          outputIndex < decode.outputsLength();
          outputIndex++) {
        int tensorIndex = decode.outputs(outputIndex);
        Tensor tensor = decode.tensors(tensorIndex);
        System.out.println(
            "graph_output="
                + tensorIndex
                + " shape="
                + shape(tensor)
                + " name="
                + tensor.name());
      }
      for (int tensorIndex = 0;
          tensorIndex < decode.tensorsLength();
          tensorIndex++) {
        Tensor tensor = decode.tensors(tensorIndex);
        if (!isDecoderBoundaryShape(tensor)) continue;
        Integer producer = producerByTensor.get(tensorIndex);
        if (producer == null) continue;
        Operator operator = decode.operators(producer);
        int builtinCode =
            model
                .operatorCodes(Math.toIntExact(operator.opcodeIndex()))
                .builtinCode();
        System.out.println(
            "boundary_tensor="
                + tensorIndex
                + " producer="
                + producer
                + " builtin="
                + builtinCode
                + " shape="
                + shape(tensor)
                + " name="
                + tensor.name());
      }
    }
  }

  private static void inspectDecoderOperators(
      Path path, int start, int endExclusive) throws Exception {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() >= 1, "Decoder subgraph is missing.");
      SubGraph decode = model.subgraphs(0);
      require(
          start >= 0 && start < endExclusive
              && endExclusive <= decode.operatorsLength(),
          "Decoder operator range is invalid.");
      for (int operatorIndex = start;
          operatorIndex < endExclusive;
          operatorIndex++) {
        Operator operator = decode.operators(operatorIndex);
        int builtinCode =
            model
                .operatorCodes(Math.toIntExact(operator.opcodeIndex()))
                .builtinCode();
        System.out.println(
            "operator="
                + operatorIndex
                + " builtin="
                + (
                    builtinCode >= 0 && builtinCode < BuiltinOperator.names.length
                        ? BuiltinOperator.name(builtinCode)
                        : "BuiltinOperator#" + builtinCode)
                + " inputs="
                + tensorVectorSummary(decode, operator, true)
                + " outputs="
                + tensorVectorSummary(decode, operator, false));
      }
    }
  }

  private static String tensorVectorSummary(
      SubGraph graph, Operator operator, boolean inputs) {
    StringBuilder result = new StringBuilder("[");
    int length = inputs ? operator.inputsLength() : operator.outputsLength();
    for (int index = 0; index < length; index++) {
      if (index > 0) result.append(", ");
      int tensorIndex = inputs ? operator.inputs(index) : operator.outputs(index);
      if (tensorIndex < 0) {
        result.append(tensorIndex);
        continue;
      }
      Tensor tensor = graph.tensors(tensorIndex);
      result
          .append(tensorIndex)
          .append(":")
          .append(shape(tensor))
          .append(":")
          .append(tensor.name());
    }
    return result.append("]").toString();
  }

  private static void inspectDecoderTensor(Path path, int tensorIndex)
      throws Exception {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      SubGraph decode = model.subgraphs(0);
      require(
          tensorIndex >= 0 && tensorIndex < decode.tensorsLength(),
          "Decoder tensor index is invalid.");
      Tensor tensor = decode.tensors(tensorIndex);
      int bufferIndex = Math.toIntExact(tensor.buffer());
      Buffer buffer = model.buffers(bufferIndex);
      System.out.println("tensor=" + tensorIndex);
      System.out.println("name=" + tensor.name());
      System.out.println("shape=" + shape(tensor));
      System.out.println("type=" + tensor.type());
      System.out.println("buffer=" + bufferIndex);
      System.out.println("buffer_data_bytes=" + buffer.dataLength());
      System.out.println("buffer_external_offset=" + readLongField(buffer, 6, 0L));
      System.out.println("buffer_external_size=" + readLongField(buffer, 8, 0L));
      QuantizationParameters quantization = tensor.quantization();
      System.out.println(
          "quantization_details_type="
              + (quantization == null ? 0 : quantization.detailsType()));
      System.out.println(
          "quantization_scale_count="
              + (quantization == null ? 0 : quantization.scaleLength()));
    }
  }

  private static void generateDecoderPrefix(
      Path source, DecoderPrefixStage stage, Path output) throws Exception {
    require(Files.isRegularFile(source), "Rewritten production model is missing.");
    require(!source.equals(output), "Prefix output must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "Decoder-prefix output must stay under a build directory.");
    require(
        Files.size(source) == REWRITTEN_PRODUCTION_SIZE,
        "Rewritten production model size changed.");
    require(
        REWRITTEN_PRODUCTION_SHA256.equals(sha256(source)),
        "Rewritten production model SHA-256 changed.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      PrefixPatchLayout layout;
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        layout = requireRewrittenPrefixContract(sourceModel, stage);
        writeInt(outputChannel, layout.operatorVector(), stage.operatorCount());
        writeInt(
            outputChannel,
            layout.graphOutputElement(),
            stage.outputTensor());
        writeInt(
            outputChannel,
            layout.signatureOutputField(),
            stage.outputTensor());
        outputChannel.force(true);
      }
      verifyDecoderPrefix(temporary, stage);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("stage=" + stage.id());
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("decode_operators=" + stage.operatorCount());
      System.out.println("decode_output_tensor=" + stage.outputTensor());
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void generateFloatKDecoderPrefix(
      Path source, DecoderPrefixStage stage, Path output) throws Exception {
    require(Files.isRegularFile(source), "Rewritten production model is missing.");
    require(
        !source.equals(output),
        "FLOAT32 K-prefix output must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "FLOAT32 K-prefix output must stay under a build directory.");
    require(
        Files.size(source) == REWRITTEN_PRODUCTION_SIZE
            && REWRITTEN_PRODUCTION_SHA256.equals(sha256(source)),
        "FLOAT32 K-prefix source is not the pinned rewritten production model.");
    require(
        stage.operatorCount() > FIRST_BLOCK_K_OPERATOR,
        "FLOAT32 K-prefix stage must include the first decoder K projection.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      long floatWeightOffset;
      long expectedSize;
      String floatWeightSha256;
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        PrefixPatchLayout prefixLayout =
            requireRewrittenPrefixContract(sourceModel, stage);
        SubGraph decode = sourceModel.subgraphs(0);
        Operator projection = decode.operators(FIRST_BLOCK_K_OPERATOR);
        OperatorCode projectionCode =
            sourceModel.operatorCodes(Math.toIntExact(projection.opcodeIndex()));
        require(
            projectionCode.builtinCode() == BuiltinOperator.FULLY_CONNECTED,
            "First decoder K projection operator changed.");
        require(
            projection.inputsLength() == 3
                && projection.inputs(0) == FIRST_BLOCK_K_INPUT_TENSOR
                && projection.inputs(1) == FIRST_BLOCK_K_WEIGHT_TENSOR
                && projection.inputs(2) == -1
                && projection.outputsLength() == 1
                && projection.outputs(0) == FIRST_BLOCK_K_OUTPUT_TENSOR,
            "First decoder K projection tensor contract changed.");

        Tensor weight = decode.tensors(FIRST_BLOCK_K_WEIGHT_TENSOR);
        require(
            weight.type() == TensorType.INT8
                && Arrays.equals(
                    tensorShape(weight),
                    new int[] {FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS})
                && weight.buffer() == FIRST_BLOCK_K_WEIGHT_BUFFER,
            "First decoder K weight tensor changed.");
        require(
            countTensorBufferReferences(decode, FIRST_BLOCK_K_WEIGHT_BUFFER) == 1,
            "First decoder K weight buffer is unexpectedly shared.");
        QuantizationParameters quantization = weight.quantization();
        require(
            quantization != null
                && quantization.detailsType() == QuantizationDetails.NONE
                && quantization.quantizedDimension() == 0
                && quantization.scaleLength() == FIRST_BLOCK_K_CHANNELS
                && (quantization.zeroPointLength() == 0
                    || quantization.zeroPointLength() == 1
                    || quantization.zeroPointLength() == FIRST_BLOCK_K_CHANNELS),
            "First decoder K per-channel quantization contract changed.");

        Buffer weightBuffer =
            sourceModel.buffers(FIRST_BLOCK_K_WEIGHT_BUFFER);
        long quantizedWeightOffset = readLongField(weightBuffer, 6, 0L);
        long quantizedWeightSize = readLongField(weightBuffer, 8, 0L);
        require(
            weightBuffer.dataLength() == 0
                && quantizedWeightOffset > 0
                && quantizedWeightSize == FIRST_BLOCK_K_WEIGHT_ELEMENTS
                && quantizedWeightOffset + quantizedWeightSize
                    <= sourceChannel.size(),
            "First decoder K external weight buffer changed.");

        ByteBuffer tensorFragment =
            buildFloatProjectionTensorFragment(weight, "k");
        long tensorFragmentOffset = align(outputChannel.size(), Integer.BYTES);
        if (tensorFragmentOffset > outputChannel.size()) {
          writeFully(
              outputChannel,
              ByteBuffer.allocate(
                  Math.toIntExact(tensorFragmentOffset - outputChannel.size())),
              outputChannel.size());
        }
        int fragmentRoot = tensorFragment.getInt(tensorFragment.position());
        require(
            fragmentRoot > 0 && fragmentRoot < tensorFragment.remaining(),
            "Generated FLOAT32 K tensor fragment root is invalid.");
        writeFully(outputChannel, tensorFragment, tensorFragmentOffset);

        int tensorVector =
            tableVectorStart(decode, 4, "decode tensor vector");
        int tensorVectorElement =
            Math.addExact(
                tensorVector + Integer.BYTES,
                Math.multiplyExact(
                    FIRST_BLOCK_K_WEIGHT_TENSOR, Integer.BYTES));
        int floatTensorTarget =
            Math.toIntExact(tensorFragmentOffset + fragmentRoot);
        writeForwardOffset(
            outputChannel,
            tensorVectorElement,
            floatTensorTarget,
            "FLOAT32 K tensor");

        floatWeightOffset = align(outputChannel.size(), 16);
        if (floatWeightOffset > outputChannel.size()) {
          writeFully(
              outputChannel,
              ByteBuffer.allocate(
                  Math.toIntExact(floatWeightOffset - outputChannel.size())),
              outputChannel.size());
        }
        ByteBuffer quantizedWeights = sourceBytes.duplicate();
        quantizedWeights.position(Math.toIntExact(quantizedWeightOffset));
        quantizedWeights.limit(
            Math.toIntExact(quantizedWeightOffset + quantizedWeightSize));
        floatWeightSha256 =
            appendDequantizedProjectionWeights(
                "k",
                outputChannel,
                floatWeightOffset,
                quantizedWeights.slice(),
                quantization);
        expectedSize =
            Math.addExact(floatWeightOffset, FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES);
        require(
            outputChannel.size() == expectedSize,
            "FLOAT32 K-prefix output size is inconsistent.");

        writeLong(
            outputChannel,
            tableFieldPosition(
                weightBuffer, 6, "K weight external offset"),
            floatWeightOffset);
        writeLong(
            outputChannel,
            tableFieldPosition(
                weightBuffer, 8, "K weight external size"),
            FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES);
        writeInt(
            outputChannel,
            prefixLayout.operatorVector(),
            stage.operatorCount());
        writeInt(
            outputChannel,
            prefixLayout.graphOutputElement(),
            stage.outputTensor());
        writeInt(
            outputChannel,
            prefixLayout.signatureOutputField(),
            stage.outputTensor());
        outputChannel.force(true);
      }
      verifyFloatKDecoderPrefix(
          temporary, stage, floatWeightOffset, expectedSize);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("variant=first-block-float-k");
      System.out.println("stage=" + stage.id());
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("float_k_weight_offset=" + floatWeightOffset);
      System.out.println("float_k_weight_bytes=" + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES);
      System.out.println("float_k_weight_sha256=" + floatWeightSha256);
      System.out.println("decode_operators=" + stage.operatorCount());
      System.out.println("decode_output_tensor=" + stage.outputTensor());
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void generateFloatSelfAttentionDecoderPrefix(
      Path source, DecoderPrefixStage stage, Path output) throws Exception {
    require(Files.isRegularFile(source), "Rewritten production model is missing.");
    require(
        !source.equals(output),
        "FLOAT32 self-attention prefix must not replace the source model.");
    require(
        hasBuildPathSegment(output),
        "FLOAT32 self-attention prefix must stay under a build directory.");
    require(
        Files.size(source) == REWRITTEN_PRODUCTION_SIZE
            && REWRITTEN_PRODUCTION_SHA256.equals(sha256(source)),
        "FLOAT32 self-attention source is not the pinned rewritten production model.");
    require(
        stage.operatorCount() > FIRST_BLOCK_K_OPERATOR,
        "FLOAT32 self-attention stage must include its K projection.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      long[] floatWeightOffsets =
          new long[FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length];
      String[] floatWeightSha256 =
          new String[FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length];
      long expectedSize;
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        PrefixPatchLayout prefixLayout =
            requireRewrittenPrefixContract(sourceModel, stage);
        SubGraph decode = sourceModel.subgraphs(0);
        ProjectionSource[] projectionSources =
            new ProjectionSource[FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length];

        for (int index = 0;
            index < FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length;
            index++) {
          FloatProjectionSpec spec =
              FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[index];
          for (int previous = 0; previous < index; previous++) {
            FloatProjectionSpec other =
                FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[previous];
            require(
                spec.weightTensor() != other.weightTensor()
                    && spec.weightBuffer() != other.weightBuffer(),
                "FLOAT32 self-attention projection constants are not unique.");
          }
          Operator projection = decode.operators(spec.operatorIndex());
          OperatorCode projectionCode =
              sourceModel.operatorCodes(
                  Math.toIntExact(projection.opcodeIndex()));
          require(
              projectionCode.builtinCode() == BuiltinOperator.FULLY_CONNECTED,
              "First decoder " + spec.label() + " projection operator changed.");
          require(
              projection.inputsLength() == 3
                  && projection.inputs(0) == spec.inputTensor()
                  && projection.inputs(1) == spec.weightTensor()
                  && projection.inputs(2) == spec.biasTensor()
                  && projection.outputsLength() == 1
                  && projection.outputs(0) == spec.outputTensor(),
              "First decoder "
                  + spec.label()
                  + " projection tensor contract changed.");

          Tensor weight = decode.tensors(spec.weightTensor());
          require(
              weight.type() == TensorType.INT8
                  && Arrays.equals(
                      tensorShape(weight),
                      new int[] {
                        FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS
                      })
                  && weight.buffer() == spec.weightBuffer(),
              "First decoder " + spec.label() + " weight tensor changed.");
          require(
              countTensorBufferReferences(decode, spec.weightBuffer()) == 1,
              "First decoder "
                  + spec.label()
                  + " weight buffer is unexpectedly shared.");
          QuantizationParameters quantization = weight.quantization();
          require(
              quantization != null
                  && quantization.detailsType() == QuantizationDetails.NONE
                  && quantization.quantizedDimension() == 0
                  && quantization.scaleLength() == FIRST_BLOCK_K_CHANNELS
                  && (quantization.zeroPointLength() == 0
                      || quantization.zeroPointLength() == 1
                      || quantization.zeroPointLength()
                          == FIRST_BLOCK_K_CHANNELS),
              "First decoder "
                  + spec.label()
                  + " per-channel quantization contract changed.");

          Buffer weightBuffer = sourceModel.buffers(spec.weightBuffer());
          long quantizedWeightOffset =
              readLongField(weightBuffer, 6, 0L);
          long quantizedWeightSize =
              readLongField(weightBuffer, 8, 0L);
          require(
              weightBuffer.dataLength() == 0
                  && quantizedWeightOffset > 0
                  && quantizedWeightSize == FIRST_BLOCK_K_WEIGHT_ELEMENTS
                  && quantizedWeightOffset + quantizedWeightSize
                      <= sourceChannel.size(),
              "First decoder "
                  + spec.label()
                  + " external weight buffer changed.");
          projectionSources[index] =
              new ProjectionSource(
                  weight,
                  weightBuffer,
                  quantization,
                  quantizedWeightOffset,
                  quantizedWeightSize);
        }

        int tensorVector =
            tableVectorStart(decode, 4, "decode tensor vector");
        for (int index = 0;
            index < FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length;
            index++) {
          FloatProjectionSpec spec =
              FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[index];
          ProjectionSource projectionSource = projectionSources[index];
          ByteBuffer tensorFragment =
              buildFloatProjectionTensorFragment(
                  projectionSource.weight(), spec.label());
          long tensorFragmentOffset =
              align(outputChannel.size(), Integer.BYTES);
          if (tensorFragmentOffset > outputChannel.size()) {
            writeFully(
                outputChannel,
                ByteBuffer.allocate(
                    Math.toIntExact(
                        tensorFragmentOffset - outputChannel.size())),
                outputChannel.size());
          }
          int fragmentRoot = tensorFragment.getInt(tensorFragment.position());
          require(
              fragmentRoot > 0 && fragmentRoot < tensorFragment.remaining(),
              "Generated FLOAT32 "
                  + spec.label()
                  + " tensor fragment root is invalid.");
          writeFully(outputChannel, tensorFragment, tensorFragmentOffset);
          int tensorVectorElement =
              Math.addExact(
                  tensorVector + Integer.BYTES,
                  Math.multiplyExact(spec.weightTensor(), Integer.BYTES));
          writeForwardOffset(
              outputChannel,
              tensorVectorElement,
              Math.toIntExact(tensorFragmentOffset + fragmentRoot),
              "FLOAT32 " + spec.label() + " tensor");
        }

        for (int index = 0;
            index < FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length;
            index++) {
          FloatProjectionSpec spec =
              FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[index];
          ProjectionSource projectionSource = projectionSources[index];
          long floatWeightOffset = align(outputChannel.size(), 16);
          if (floatWeightOffset > outputChannel.size()) {
            writeFully(
                outputChannel,
                ByteBuffer.allocate(
                    Math.toIntExact(
                        floatWeightOffset - outputChannel.size())),
                outputChannel.size());
          }
          ByteBuffer quantizedWeights = sourceBytes.duplicate();
          quantizedWeights.position(
              Math.toIntExact(projectionSource.quantizedWeightOffset()));
          quantizedWeights.limit(
              Math.toIntExact(
                  projectionSource.quantizedWeightOffset()
                      + projectionSource.quantizedWeightSize()));
          floatWeightOffsets[index] = floatWeightOffset;
          floatWeightSha256[index] =
              appendDequantizedProjectionWeights(
                  spec.label(),
                  outputChannel,
                  floatWeightOffset,
                  quantizedWeights.slice(),
                  projectionSource.quantization());
          require(
              outputChannel.size()
                  == floatWeightOffset + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES,
              "FLOAT32 "
                  + spec.label()
                  + " output size is inconsistent.");
          writeLong(
              outputChannel,
              tableFieldPosition(
                  projectionSource.weightBuffer(),
                  6,
                  spec.label() + " weight external offset"),
              floatWeightOffset);
          writeLong(
              outputChannel,
              tableFieldPosition(
                  projectionSource.weightBuffer(),
                  8,
                  spec.label() + " weight external size"),
              FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES);
        }
        expectedSize = outputChannel.size();
        writeInt(
            outputChannel,
            prefixLayout.operatorVector(),
            stage.operatorCount());
        writeInt(
            outputChannel,
            prefixLayout.graphOutputElement(),
            stage.outputTensor());
        writeInt(
            outputChannel,
            prefixLayout.signatureOutputField(),
            stage.outputTensor());
        outputChannel.force(true);
      }

      verifyFloatSelfAttentionDecoderPrefix(
          temporary, stage, floatWeightOffsets, expectedSize);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("variant=first-block-float-self-attention");
      System.out.println("stage=" + stage.id());
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      for (int index = 0;
          index < FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length;
          index++) {
        FloatProjectionSpec spec =
            FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[index];
        System.out.println(
            "float_"
                + spec.label()
                + "_weight_offset="
                + floatWeightOffsets[index]);
        System.out.println(
            "float_"
                + spec.label()
                + "_weight_sha256="
                + floatWeightSha256[index]);
      }
      System.out.println(
          "float_projection_weight_bytes=" + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES);
      System.out.println("decode_operators=" + stage.operatorCount());
      System.out.println("decode_output_tensor=" + stage.outputTensor());
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void generateMaskRankReshapesVerified(
      Path source, String expectedSha256, Path output) throws Exception {
    require(Files.isRegularFile(source), "Explicit-key source model is missing.");
    require(!source.equals(output), "Mask-rank output must not replace its source.");
    require(
        hasBuildPathSegment(output),
        "Mask-rank output must stay under a build directory.");
    requireSha256(expectedSha256);
    String actualSha256 = sha256(source);
    require(
        expectedSha256.equals(actualSha256),
        "Source SHA-256 changed: expected "
            + expectedSha256
            + ", found "
            + actualSha256
            + ".");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      int decodeIndex;
      int sourceTensorCount;
      int sourceOperatorCount;
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        SignatureDef decodeSignature = requireSignature(sourceModel, "decode");
        // The cached 0.2.0 schema predates the generated subgraphIndex()
        // accessor, but the current SignatureDef wire field is stable.
        decodeIndex = readIntField(decodeSignature, 10, 0);
        require(
            decodeIndex >= 0 && decodeIndex < sourceModel.subgraphsLength(),
            "Decode signature subgraph index is invalid.");
        SubGraph decode = sourceModel.subgraphs(decodeIndex);
        validateCloneableGraph(decode);
        List<Integer> maskAdds =
            discoverVerifiedMaskAdds(sourceModel, decode);
        sourceTensorCount = decode.tensorsLength();
        sourceOperatorCount = decode.operatorsLength();
        ByteBuffer decodeFragment =
            buildVerifiedMaskRankSubgraph(
                sourceModel, decode, maskAdds);
        long fragmentOffset = align(outputChannel.size(), Integer.BYTES);
        if (fragmentOffset > outputChannel.size()) {
          writeFully(
              outputChannel,
              ByteBuffer.allocate(
                  Math.toIntExact(fragmentOffset - outputChannel.size())),
              outputChannel.size());
        }
        int fragmentRoot = decodeFragment.getInt(decodeFragment.position());
        require(
            fragmentRoot > 0 && fragmentRoot < decodeFragment.remaining(),
            "Generated mask-rank subgraph root is invalid.");
        writeFully(outputChannel, decodeFragment, fragmentOffset);
        int subgraphVector =
            tableVectorStart(sourceModel, 8, "model subgraphs");
        int vectorElement =
            Math.addExact(
                subgraphVector + Integer.BYTES,
                Math.multiplyExact(decodeIndex, Integer.BYTES));
        writeForwardOffset(
            outputChannel,
            vectorElement,
            Math.toIntExact(fragmentOffset + fragmentRoot),
            "hash-verified mask-rank decode subgraph");
        outputChannel.force(true);
      }
      verifyMaskRankReshapes(
          source,
          temporary,
          decodeIndex,
          sourceTensorCount,
          sourceOperatorCount);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("source_contract=hash-verified FP16 experiment");
      System.out.println("variant=explicit-key-plus-mask-rank-reshape");
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("mask_rank_reshapes=" + FULL_DECODER_MASK_RESHAPE_COUNT);
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static List<Integer> discoverVerifiedMaskAdds(
      Model model, SubGraph decode) {
    List<Integer> result = new ArrayList<>();
    for (int index = 1; index + 1 < decode.operatorsLength(); index++) {
      Operator add = decode.operators(index);
      if (model
                  .operatorCodes(Math.toIntExact(add.opcodeIndex()))
                  .builtinCode()
              != BuiltinOperator.ADD
          || add.inputsLength() != 2
          || add.outputsLength() != 1) {
        continue;
      }
      Tensor scores = decode.tensors(add.inputs(0));
      Tensor mask = decode.tensors(add.inputs(1));
      Tensor masked = decode.tensors(add.outputs(0));
      if (!Arrays.equals(tensorShape(scores), new int[] {20, 128, 128})
          || !Arrays.equals(tensorShape(mask), new int[] {1, 1, 128, 128})
          || !Arrays.equals(
              tensorShape(masked), new int[] {1, 20, 128, 128})) {
        continue;
      }
      Operator producer = decode.operators(index - 1);
      require(
          model
                      .operatorCodes(Math.toIntExact(producer.opcodeIndex()))
                      .builtinCode()
                  == BuiltinOperator.BATCH_MATMUL
              && producer.outputsLength() == 1
              && producer.outputs(0) == add.inputs(0),
          "Causal-mask ADD score producer changed at operator " + index + ".");
      result.add(index);
    }
    require(
        result.size() == FULL_DECODER_MASK_RESHAPE_COUNT,
        "Expected "
            + FULL_DECODER_MASK_RESHAPE_COUNT
            + " decoder causal-mask ADD sites, found "
            + result
            + ".");
    return result;
  }

  private static ByteBuffer buildVerifiedMaskRankSubgraph(
      Model model, SubGraph decode, List<Integer> maskAdds) {
    Map<Integer, Integer> rankTensorForAdd = new LinkedHashMap<>();
    for (int index = 0; index < maskAdds.size(); index++) {
      rankTensorForAdd.put(
          maskAdds.get(index), decode.tensorsLength() + index);
    }

    FlatBufferBuilder builder = new FlatBufferBuilder(24 * 1024 * 1024);
    int[] tensorTables =
        new int[decode.tensorsLength() + maskAdds.size()];
    for (int index = 0; index < decode.tensorsLength(); index++) {
      tensorTables[index] = cloneTensor(builder, decode.tensors(index));
    }
    for (Map.Entry<Integer, Integer> entry : rankTensorForAdd.entrySet()) {
      Operator add = decode.operators(entry.getKey());
      Tensor scores = decode.tensors(add.inputs(0));
      tensorTables[entry.getValue()] =
          createTensor(
              builder,
              new int[] {1, 20, 128, 128},
              TensorType.FLOAT32,
              0,
              scores.name() + "/voicememo_explicit_rank4",
              0,
              false,
              null,
              true,
              true);
    }
    int tensors = SubGraph.createTensorsVector(builder, tensorTables);

    int reshapeOpcode = findOperatorCode(model, BuiltinOperator.RESHAPE);
    require(reshapeOpcode >= 0, "Source model has no RESHAPE operator code.");
    int[] operatorTables =
        new int[decode.operatorsLength() + maskAdds.size()];
    int destination = 0;
    for (int index = 0; index < decode.operatorsLength(); index++) {
      Operator source = decode.operators(index);
      Integer rankTensor = rankTensorForAdd.get(index);
      if (rankTensor != null) {
        int rankShape =
            ReshapeOptions.createNewShapeVector(
                builder, new int[] {1, 20, 128, 128});
        int rankOptions =
            ReshapeOptions.createReshapeOptions(builder, rankShape);
        operatorTables[destination++] =
            createOperator(
                builder,
                reshapeOpcode,
                new int[] {source.inputs(0)},
                new int[] {rankTensor},
                BuiltinOptions.ReshapeOptions,
                rankOptions,
                new byte[0],
                (byte) 0,
                new boolean[0],
                new int[0]);
        operatorTables[destination++] =
            cloneOperatorReplacingInput(builder, source, 0, rankTensor);
      } else {
        operatorTables[destination++] =
            cloneOperator(builder, source, null);
      }
    }
    require(
        destination == operatorTables.length,
        "Mask-rank decoder operator count is inconsistent.");
    int operators = SubGraph.createOperatorsVector(builder, operatorTables);
    int inputs =
        SubGraph.createInputsVector(builder, operatorInputs(decode, true));
    int outputs =
        SubGraph.createOutputsVector(builder, operatorInputs(decode, false));
    int name =
        builder.createString(
            decode.name() == null
                ? "decode_voicememo_mask_rank"
                : decode.name() + "_voicememo_mask_rank");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, inputs, outputs, operators, name);
    builder.finish(graph);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void verifyMaskRankReshapes(
      Path source,
      Path output,
      int decodeIndex,
      int sourceTensorCount,
      int sourceOperatorCount)
      throws Exception {
    require(
        Files.size(output) > Files.size(source),
        "Mask-rank model did not grow.");
    try (FileChannel sourceChannel =
            FileChannel.open(source, StandardOpenOption.READ);
        FileChannel outputChannel =
            FileChannel.open(output, StandardOpenOption.READ)) {
      Model before =
          parseModel(
              sourceChannel
                  .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                  .order(ByteOrder.LITTLE_ENDIAN));
      Model after =
          parseModel(
              outputChannel
                  .map(FileChannel.MapMode.READ_ONLY, 0, outputChannel.size())
                  .order(ByteOrder.LITTLE_ENDIAN));
      require(
          before.subgraphsLength() == after.subgraphsLength(),
          "Mask-rank subgraph count changed.");
      SubGraph beforeDecode = before.subgraphs(decodeIndex);
      SubGraph afterDecode = after.subgraphs(decodeIndex);
      require(
          afterDecode.tensorsLength()
                  == sourceTensorCount + FULL_DECODER_MASK_RESHAPE_COUNT
              && afterDecode.operatorsLength()
                  == sourceOperatorCount + FULL_DECODER_MASK_RESHAPE_COUNT,
          "Mask-rank decode graph size changed unexpectedly.");
      require(
          Arrays.equals(
                  operatorInputs(beforeDecode, true),
                  operatorInputs(afterDecode, true))
              && Arrays.equals(
                  operatorInputs(beforeDecode, false),
                  operatorInputs(afterDecode, false)),
          "Mask-rank decode signature tensors changed.");
      int barriers = 0;
      for (int index = 0;
          index < afterDecode.operatorsLength();
          index++) {
        Operator operator = afterDecode.operators(index);
        if (after
                    .operatorCodes(Math.toIntExact(operator.opcodeIndex()))
                    .builtinCode()
                != BuiltinOperator.RESHAPE
            || operator.outputsLength() != 1
            || operator.outputs(0) < sourceTensorCount) {
          continue;
        }
        require(
            operator.inputsLength() == 1
                && Arrays.equals(
                    tensorShape(afterDecode.tensors(operator.inputs(0))),
                    new int[] {20, 128, 128})
                && Arrays.equals(
                    tensorShape(afterDecode.tensors(operator.outputs(0))),
                    new int[] {1, 20, 128, 128})
                && index + 1 < afterDecode.operatorsLength()
                && afterDecode.operators(index + 1).inputs(0)
                    == operator.outputs(0),
            "Mask-rank barrier changed.");
        barriers++;
      }
      require(
          barriers == FULL_DECODER_MASK_RESHAPE_COUNT,
          "Mask-rank barrier count changed: " + barriers + ".");
      requireNoImplicitAdjY(
          after, afterDecode, afterDecode.operatorsLength());
    }
  }

  /**
   * Breaks the GPU compiler's BMM-to-broadcast-ADD layout path by making the
   * score rank explicit before the causal mask is added.
   *
   * <p>The source is the pinned first-block FLOAT32 self-attention diagnostic
   * through the complete first self-attention residual. Only its decode
   * subgraph is redirected to a prefix containing:
   * BMM [20,128,128], RESHAPE [1,20,128,128], then broadcast ADD.
   */
  private static void generateMaskAddRankReshapeDecoderPrefix(
      Path source, Path output) throws Exception {
    require(Files.isRegularFile(source), "FLOAT32 self-attention source is missing.");
    require(
        !source.equals(output),
        "Mask-ADD rank-reshape prefix must not replace its source.");
    require(
        hasBuildPathSegment(output),
        "Mask-ADD rank-reshape prefix must stay under a build directory.");
    require(
        Files.size(source) == FLOAT_SELF_ATTENTION_SOURCE_SIZE
            && FLOAT_SELF_ATTENTION_SOURCE_SHA256.equals(sha256(source)),
        "Mask-ADD rank-reshape source is not the pinned FLOAT32 self-attention model.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        ByteBuffer decodeFragment =
            buildMaskAddRankReshapeSubgraph(sourceModel);
        long fragmentOffset = align(outputChannel.size(), Integer.BYTES);
        if (fragmentOffset > outputChannel.size()) {
          writeFully(
              outputChannel,
              ByteBuffer.allocate(
                  Math.toIntExact(fragmentOffset - outputChannel.size())),
              outputChannel.size());
        }
        int fragmentRoot = decodeFragment.getInt(decodeFragment.position());
        require(
            fragmentRoot > 0 && fragmentRoot < decodeFragment.remaining(),
            "Generated mask-ADD rank-reshape subgraph root is invalid.");
        writeFully(outputChannel, decodeFragment, fragmentOffset);

        int subgraphVector =
            tableVectorStart(sourceModel, 8, "model subgraphs");
        writeForwardOffset(
            outputChannel,
            subgraphVector + Integer.BYTES,
            Math.toIntExact(fragmentOffset + fragmentRoot),
            "mask-ADD rank-reshape decode subgraph");
        SignatureDef signature = requireSignature(sourceModel, "decode");
        require(
            signature.outputsLength() == 1,
            "FLOAT32 self-attention decode signature output count changed.");
        writeInt(
            outputChannel,
            tableFieldPosition(
                signature.outputs(0),
                6,
                "mask-ADD rank-reshape signature output"),
            144);
        outputChannel.force(true);
      }
      verifyMaskAddRankReshapeDecoderPrefix(temporary);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println(
          "variant=first-block-float-self-attention-mask-rank-reshape");
      System.out.println("stage=after-self-mask-add");
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println("decode_operators=32");
      System.out.println("decode_output_tensor=144");
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static ByteBuffer buildMaskAddRankReshapeSubgraph(Model model) {
    require(model.subgraphsLength() == 2, "FLOAT32 source subgraph count changed.");
    SubGraph decode = model.subgraphs(0);
    require(
        decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT
            && decode.operatorsLength() == 43
            && decode.outputsLength() == 1
            && decode.outputs(0) == 155,
        "FLOAT32 self-attention source decode contract changed.");
    require(
        Arrays.equals(
                tensorShape(decode.tensors(142)),
                new int[] {20, 128, 128})
            && Arrays.equals(
                tensorShape(decode.tensors(143)),
                new int[] {1, 20, 128, 128})
            && Arrays.equals(
                tensorShape(decode.tensors(144)),
                new int[] {20, 128, 128}),
        "Mask-ADD score tensor shapes changed.");
    Operator score = decode.operators(29);
    Operator add = decode.operators(30);
    Operator reshape = decode.operators(31);
    require(
        model.operatorCodes(Math.toIntExact(score.opcodeIndex())).builtinCode()
                == BuiltinOperator.BATCH_MATMUL
            && score.inputsLength() == 2
            && score.outputsLength() == 1
            && score.outputs(0) == 142,
        "Mask-ADD score BMM contract changed.");
    require(
        model.operatorCodes(Math.toIntExact(add.opcodeIndex())).builtinCode()
                == BuiltinOperator.ADD
            && add.inputsLength() == 2
            && add.inputs(0) == 142
            && add.inputs(1) == 42
            && add.outputsLength() == 1
            && add.outputs(0) == 143,
        "Mask-ADD operator contract changed.");
    require(
        model.operatorCodes(Math.toIntExact(reshape.opcodeIndex())).builtinCode()
                == BuiltinOperator.RESHAPE
            && reshape.outputsLength() == 1
            && reshape.outputs(0) == 144,
        "Post-mask reshape contract changed.");

    FlatBufferBuilder builder = new FlatBufferBuilder(16 * 1024 * 1024);
    int[] tensorTables = new int[decode.tensorsLength()];
    for (int index = 0; index < decode.tensorsLength(); index++) {
      Tensor tensor = decode.tensors(index);
      tensorTables[index] =
          index == 144
              ? cloneTensorWithShape(
                  builder, tensor, new int[] {1, 20, 128, 128})
              : cloneTensor(builder, tensor);
    }
    int tensors = SubGraph.createTensorsVector(builder, tensorTables);

    int[] operatorTables = new int[32];
    for (int index = 0; index < 30; index++) {
      operatorTables[index] = cloneOperator(builder, decode.operators(index), null);
    }
    int rankShape =
        ReshapeOptions.createNewShapeVector(
            builder, new int[] {1, 20, 128, 128});
    int rankOptions =
        ReshapeOptions.createReshapeOptions(builder, rankShape);
    operatorTables[30] =
        createOperator(
            builder,
            Math.toIntExact(reshape.opcodeIndex()),
            new int[] {142},
            new int[] {143},
            BuiltinOptions.ReshapeOptions,
            rankOptions,
            new byte[0],
            (byte) 0,
            new boolean[0],
            new int[0]);
    operatorTables[31] =
        createOperator(
            builder,
            Math.toIntExact(add.opcodeIndex()),
            new int[] {143, 42},
            new int[] {144},
            add.builtinOptionsType(),
            cloneBuiltinOptions(builder, add, false),
            new byte[0],
            add.customOptionsFormat(),
            new boolean[0],
            new int[0]);
    int operators = SubGraph.createOperatorsVector(builder, operatorTables);
    int inputs =
        SubGraph.createInputsVector(builder, operatorInputs(decode, true));
    int outputs =
        SubGraph.createOutputsVector(builder, new int[] {144});
    int name =
        builder.createString("decode_mask_add_rank_reshape_prefix");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, inputs, outputs, operators, name);
    builder.finish(graph);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void verifyMaskAddRankReshapeDecoderPrefix(Path output)
      throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 2, "Rank-reshape prefix subgraph count changed.");
      SubGraph decode = model.subgraphs(0);
      require(
          decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT
              && decode.operatorsLength() == 32
              && decode.outputsLength() == 1
              && decode.outputs(0) == 144,
          "Rank-reshape prefix decode contract changed.");
      require(
          Arrays.equals(
              tensorShape(decode.tensors(144)),
              new int[] {1, 20, 128, 128}),
          "Rank-reshape prefix output tensor shape changed.");
      Operator reshape = decode.operators(30);
      ReshapeOptions reshapeOptions =
          (ReshapeOptions)
              reshape.builtinOptions(new ReshapeOptions());
      require(
          model.operatorCodes(Math.toIntExact(reshape.opcodeIndex())).builtinCode()
                  == BuiltinOperator.RESHAPE
              && reshape.inputsLength() == 1
              && reshape.inputs(0) == 142
              && reshape.outputsLength() == 1
              && reshape.outputs(0) == 143
              && reshapeOptions != null
              && reshapeOptions.newShapeLength() == 4
              && reshapeOptions.newShape(0) == 1
              && reshapeOptions.newShape(1) == 20
              && reshapeOptions.newShape(2) == 128
              && reshapeOptions.newShape(3) == 128,
          "Explicit pre-mask rank reshape changed.");
      Operator add = decode.operators(31);
      require(
          model.operatorCodes(Math.toIntExact(add.opcodeIndex())).builtinCode()
                  == BuiltinOperator.ADD
              && add.inputsLength() == 2
              && add.inputs(0) == 143
              && add.inputs(1) == 42
              && add.outputsLength() == 1
              && add.outputs(0) == 144,
          "Rank-reshape causal-mask ADD changed.");
      SignatureDef signature = requireSignature(model, "decode");
      require(
          signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 144,
          "Rank-reshape prefix signature changed.");
      requireNoImplicitAdjY(model, decode, decode.operatorsLength());
    }
  }

  private static void generateFullFloatDecoder(Path source, Path output)
      throws Exception {
    require(Files.isRegularFile(source), "Rewritten production model is missing.");
    require(!source.equals(output), "Full FLOAT32 decoder must not replace its source.");
    require(
        hasBuildPathSegment(output),
        "Full FLOAT32 decoder output must stay under a build directory.");
    require(
        Files.size(source) == REWRITTEN_PRODUCTION_SIZE
            && REWRITTEN_PRODUCTION_SHA256.equals(sha256(source)),
        "Full FLOAT32 decoder source is not the pinned explicit-key model.");
    Files.createDirectories(output.getParent());

    Path temporary = output.resolveSibling(output.getFileName() + ".tmp");
    Files.deleteIfExists(temporary);
    try {
      Files.copy(source, temporary, StandardCopyOption.REPLACE_EXISTING);
      Map<Integer, Long> floatOffsets = new LinkedHashMap<>();
      Map<Integer, Long> floatSizes = new LinkedHashMap<>();
      long expectedSize;
      long totalFloatWeightBytes = 0L;
      try (FileChannel sourceChannel =
              FileChannel.open(source, StandardOpenOption.READ);
          FileChannel outputChannel =
              FileChannel.open(
                  temporary, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            sourceChannel
                .map(FileChannel.MapMode.READ_ONLY, 0, sourceChannel.size())
                .order(ByteOrder.LITTLE_ENDIAN);
        Model sourceModel = parseModel(sourceBytes);
        require(
            sourceModel.subgraphsLength() == 2,
            "Full decoder source subgraph count changed.");
        SubGraph decode = sourceModel.subgraphs(0);
        require(
            decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT
                && decode.operatorsLength() == REWRITTEN_DECODER_OPERATOR_COUNT
                && decode.outputsLength() == 1
                && decode.outputs(0) == REWRITTEN_LOGITS_TENSOR,
            "Full decoder source graph contract changed.");
        require(
            sourceModel.subgraphs(1).operatorsLength()
                == REWRITTEN_ENCODER_OPERATOR_COUNT,
            "Full decoder source encoder changed.");
        List<FullDecoderProjection> projections =
            discoverFullDecoderProjections(sourceModel, decode, sourceChannel.size());
        List<Integer> maskAdds =
            discoverFullDecoderMaskAdds(sourceModel, decode);

        ByteBuffer decodeFragment =
            buildFullFloatDecoderSubgraph(
                sourceModel, decode, projections, maskAdds);
        long fragmentOffset = align(outputChannel.size(), Integer.BYTES);
        if (fragmentOffset > outputChannel.size()) {
          writeFully(
              outputChannel,
              ByteBuffer.allocate(
                  Math.toIntExact(fragmentOffset - outputChannel.size())),
              outputChannel.size());
        }
        int fragmentRoot = decodeFragment.getInt(decodeFragment.position());
        require(
            fragmentRoot > 0 && fragmentRoot < decodeFragment.remaining(),
            "Generated full FLOAT32 decoder subgraph root is invalid.");
        writeFully(outputChannel, decodeFragment, fragmentOffset);
        int subgraphVector =
            tableVectorStart(sourceModel, 8, "model subgraphs");
        writeForwardOffset(
            outputChannel,
            subgraphVector + Integer.BYTES,
            Math.toIntExact(fragmentOffset + fragmentRoot),
            "full FLOAT32 decoder subgraph");

        for (FullDecoderProjection projection : projections) {
          long floatOffset = align(outputChannel.size(), 16);
          if (floatOffset > outputChannel.size()) {
            writeFully(
                outputChannel,
                ByteBuffer.allocate(
                    Math.toIntExact(floatOffset - outputChannel.size())),
                outputChannel.size());
          }
          ByteBuffer quantized = sourceBytes.duplicate();
          quantized.position(Math.toIntExact(projection.quantizedOffset()));
          quantized.limit(
              Math.toIntExact(
                  projection.quantizedOffset() + projection.quantizedSize()));
          appendDequantizedFullyConnectedWeights(
              projection,
              outputChannel,
              floatOffset,
              quantized.slice());
          long floatBytes =
              Math.multiplyExact(projection.elementCount(), (long) Float.BYTES);
          require(
              outputChannel.size() == floatOffset + floatBytes,
              "Full FLOAT32 decoder weight append size is inconsistent.");
          writeLong(
              outputChannel,
              tableFieldPosition(
                  projection.buffer(),
                  6,
                  projection.label() + " external offset"),
              floatOffset);
          writeLong(
              outputChannel,
              tableFieldPosition(
                  projection.buffer(),
                  8,
                  projection.label() + " external size"),
              floatBytes);
          floatOffsets.put(projection.weightTensor(), floatOffset);
          floatSizes.put(projection.weightTensor(), floatBytes);
          totalFloatWeightBytes =
              Math.addExact(totalFloatWeightBytes, floatBytes);
        }
        expectedSize = outputChannel.size();
        outputChannel.force(true);
      }

      verifyFullFloatDecoder(
          temporary, expectedSize, floatOffsets, floatSizes);
      moveAtomically(temporary, output);
      System.out.println("source=" + source);
      System.out.println("variant=full-float-decoder-mask-rank-reshape");
      System.out.println("output=" + output);
      System.out.println("bytes=" + Files.size(output));
      System.out.println("sha256=" + sha256(output));
      System.out.println(
          "float_projection_count=" + floatOffsets.size());
      System.out.println(
          "float_projection_weight_bytes=" + totalFloatWeightBytes);
      System.out.println(
          "causal_mask_rank_reshape_count="
              + FULL_DECODER_MASK_RESHAPE_COUNT);
      System.out.println(
          "decode_operators=" + FULL_DECODER_REWRITTEN_OPERATOR_COUNT);
      System.out.println("remaining_adj_y_true=0");
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static List<FullDecoderProjection> discoverFullDecoderProjections(
      Model model, SubGraph decode, long sourceSize) {
    List<FullDecoderProjection> projections = new ArrayList<>();
    Map<Integer, Integer> operatorForWeight = new LinkedHashMap<>();
    Map<Integer, Integer> tensorForBuffer = new LinkedHashMap<>();
    for (int operatorIndex = 0;
        operatorIndex < decode.operatorsLength();
        operatorIndex++) {
      Operator operator = decode.operators(operatorIndex);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(operator.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.FULLY_CONNECTED) continue;
      require(
          operator.inputsLength() >= 2,
          "Decoder FULLY_CONNECTED input count changed at " + operatorIndex + ".");
      int weightTensor = operator.inputs(1);
      require(weightTensor >= 0, "Decoder projection has no weight tensor.");
      Tensor weight = decode.tensors(weightTensor);
      require(
          weight.type() == TensorType.INT8 && weight.shapeLength() == 2,
          "Decoder projection weight is not rank-2 INT8 at operator "
              + operatorIndex
              + ".");
      int bufferIndex = Math.toIntExact(weight.buffer());
      require(
          operatorForWeight.put(weightTensor, operatorIndex) == null,
          "Decoder projection weight tensor is unexpectedly shared: "
              + weightTensor
              + ".");
      require(
          tensorForBuffer.put(bufferIndex, weightTensor) == null,
          "Decoder projection weight buffer is unexpectedly shared: "
              + bufferIndex
              + ".");
      require(
          countModelTensorBufferReferences(model, bufferIndex) == 1,
          "Decoder projection buffer has multiple model references: "
              + bufferIndex
              + ".");
      QuantizationParameters quantization = weight.quantization();
      int outputChannels = weight.shape(0);
      require(
          quantization != null
              && quantization.detailsType() == QuantizationDetails.NONE
              && quantization.quantizedDimension() == 0
              && quantization.scaleLength() == outputChannels
              && (
                  quantization.zeroPointLength() == 0
                      || quantization.zeroPointLength() == 1
                      || quantization.zeroPointLength() == outputChannels),
          "Decoder projection quantization changed at operator "
              + operatorIndex
              + ".");
      long elementCount =
          Math.multiplyExact((long) weight.shape(0), (long) weight.shape(1));
      Buffer buffer = model.buffers(bufferIndex);
      long quantizedOffset = readLongField(buffer, 6, 0L);
      long quantizedSize = readLongField(buffer, 8, 0L);
      require(
          buffer.dataLength() == 0
              && quantizedOffset > 0
              && quantizedSize == elementCount
              && quantizedOffset + quantizedSize <= sourceSize,
          "Decoder projection external buffer changed at operator "
              + operatorIndex
              + ".");
      projections.add(
          new FullDecoderProjection(
              operatorIndex,
              weightTensor,
              bufferIndex,
              weight,
              buffer,
              quantization,
              quantizedOffset,
              quantizedSize,
              outputChannels,
              Math.toIntExact(elementCount / outputChannels),
              elementCount,
              "decoder_fc_" + operatorIndex + "_tensor_" + weightTensor));
    }
    require(
        projections.size() == FULL_DECODER_FLOAT_PROJECTION_COUNT,
        "Decoder FLOAT32 projection count changed: " + projections.size() + ".");
    return projections;
  }

  private static List<Integer> discoverFullDecoderMaskAdds(
      Model model, SubGraph decode) {
    List<Integer> result = new ArrayList<>();
    for (int index = 1; index + 1 < decode.operatorsLength(); index++) {
      Operator add = decode.operators(index);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(add.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.ADD
          || add.inputsLength() != 2
          || add.outputsLength() != 1) {
        continue;
      }
      Tensor scores = decode.tensors(add.inputs(0));
      Tensor mask = decode.tensors(add.inputs(1));
      Tensor masked = decode.tensors(add.outputs(0));
      if (!Arrays.equals(tensorShape(scores), new int[] {20, 128, 128})
          || !Arrays.equals(tensorShape(mask), new int[] {1, 1, 128, 128})
          || !Arrays.equals(
              tensorShape(masked), new int[] {1, 20, 128, 128})) {
        continue;
      }
      Operator producer = decode.operators(index - 1);
      Operator consumer = decode.operators(index + 1);
      require(
          model.operatorCodes(Math.toIntExact(producer.opcodeIndex())).builtinCode()
                  == BuiltinOperator.BATCH_MATMUL
              && producer.outputsLength() == 1
              && producer.outputs(0) == add.inputs(0),
          "Causal-mask ADD score producer changed at operator " + index + ".");
      require(
          model.operatorCodes(Math.toIntExact(consumer.opcodeIndex())).builtinCode()
                  == BuiltinOperator.RESHAPE
              && consumer.inputsLength() == 2
              && consumer.inputs(0) == add.outputs(0),
          "Causal-mask ADD reshape consumer changed at operator " + index + ".");
      result.add(index);
    }
    require(
        result.equals(List.of(30, 109, 188, 267)),
        "Decoder causal-mask ADD sites changed: " + result + ".");
    return result;
  }

  private static ByteBuffer buildFullFloatDecoderSubgraph(
      Model model,
      SubGraph decode,
      List<FullDecoderProjection> projections,
      List<Integer> maskAdds) {
    Map<Integer, FullDecoderProjection> projectionForTensor =
        new LinkedHashMap<>();
    for (FullDecoderProjection projection : projections) {
      projectionForTensor.put(projection.weightTensor(), projection);
    }
    Map<Integer, Integer> rankTensorForAdd = new LinkedHashMap<>();
    for (int index = 0; index < maskAdds.size(); index++) {
      rankTensorForAdd.put(
          maskAdds.get(index), REWRITTEN_DECODER_TENSOR_COUNT + index);
    }

    FlatBufferBuilder builder = new FlatBufferBuilder(24 * 1024 * 1024);
    int[] tensorTables = new int[FULL_DECODER_REWRITTEN_TENSOR_COUNT];
    for (int index = 0; index < decode.tensorsLength(); index++) {
      Tensor tensor = decode.tensors(index);
      tensorTables[index] =
          projectionForTensor.containsKey(index)
              ? cloneTensorAsFloat(builder, tensor)
              : cloneTensor(builder, tensor);
    }
    for (Map.Entry<Integer, Integer> entry : rankTensorForAdd.entrySet()) {
      int addIndex = entry.getKey();
      int tensorIndex = entry.getValue();
      Tensor scores = decode.tensors(decode.operators(addIndex).inputs(0));
      tensorTables[tensorIndex] =
          createTensor(
              builder,
              new int[] {1, 20, 128, 128},
              TensorType.FLOAT32,
              0,
              scores.name() + "/voicememo_explicit_rank4",
              0,
              false,
              null,
              true,
              true);
    }
    int tensors = SubGraph.createTensorsVector(builder, tensorTables);

    int[] operatorTables = new int[FULL_DECODER_REWRITTEN_OPERATOR_COUNT];
    int destination = 0;
    for (int index = 0; index < decode.operatorsLength(); index++) {
      Operator source = decode.operators(index);
      Integer rankTensor = rankTensorForAdd.get(index);
      if (rankTensor != null) {
        Operator postAddReshape = decode.operators(index + 1);
        int rankShape =
            ReshapeOptions.createNewShapeVector(
                builder, new int[] {1, 20, 128, 128});
        int rankOptions =
            ReshapeOptions.createReshapeOptions(builder, rankShape);
        operatorTables[destination++] =
            createOperator(
                builder,
                Math.toIntExact(postAddReshape.opcodeIndex()),
                new int[] {source.inputs(0)},
                new int[] {rankTensor},
                BuiltinOptions.ReshapeOptions,
                rankOptions,
                new byte[0],
                (byte) 0,
                new boolean[0],
                new int[0]);
        operatorTables[destination++] =
            cloneOperatorReplacingInput(builder, source, 0, rankTensor);
      } else {
        operatorTables[destination++] =
            cloneOperator(builder, source, null);
      }
    }
    require(
        destination == operatorTables.length,
        "Full FLOAT32 decoder operator count is inconsistent.");
    int operators = SubGraph.createOperatorsVector(builder, operatorTables);
    int inputs =
        SubGraph.createInputsVector(builder, operatorInputs(decode, true));
    int outputs =
        SubGraph.createOutputsVector(builder, operatorInputs(decode, false));
    int name = builder.createString("decode_full_float_mask_rank_reshape");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, inputs, outputs, operators, name);
    builder.finish(graph);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void appendDequantizedFullyConnectedWeights(
      FullDecoderProjection projection,
      FileChannel output,
      long outputOffset,
      ByteBuffer quantizedWeights)
      throws Exception {
    require(
        quantizedWeights.remaining() == projection.elementCount(),
        "Quantized decoder projection byte count changed for "
            + projection.label()
            + ".");
    ByteBuffer floats =
        ByteBuffer.allocateDirect(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
    long writeOffset = outputOffset;
    QuantizationParameters quantization = projection.quantization();
    for (long index = 0; index < projection.elementCount(); index++) {
      int channel =
          Math.toIntExact(index / projection.innerElements());
      long zeroPoint;
      if (quantization.zeroPointLength() == 0) {
        zeroPoint = 0L;
      } else if (quantization.zeroPointLength() == 1) {
        zeroPoint = quantization.zeroPoint(0);
      } else {
        zeroPoint = quantization.zeroPoint(channel);
      }
      float dequantized =
          (quantizedWeights.get(Math.toIntExact(index)) - zeroPoint)
              * quantization.scale(channel);
      if (floats.remaining() < Float.BYTES) {
        floats.flip();
        int bytes = floats.remaining();
        writeFully(output, floats, writeOffset);
        writeOffset += bytes;
        floats.clear();
      }
      floats.putFloat(dequantized);
    }
    floats.flip();
    int finalBytes = floats.remaining();
    writeFully(output, floats, writeOffset);
    writeOffset += finalBytes;
    require(
        writeOffset
            == outputOffset
                + Math.multiplyExact(
                    projection.elementCount(), (long) Float.BYTES),
        "FLOAT32 decoder projection byte count is inconsistent for "
            + projection.label()
            + ".");
  }

  private static void verifyFullFloatDecoder(
      Path output,
      long expectedSize,
      Map<Integer, Long> expectedOffsets,
      Map<Integer, Long> expectedSizes)
      throws Exception {
    require(
        Files.size(output) == expectedSize,
        "Full FLOAT32 decoder model size changed.");
    require(
        expectedOffsets.size() == FULL_DECODER_FLOAT_PROJECTION_COUNT
            && expectedSizes.keySet().equals(expectedOffsets.keySet()),
        "Full FLOAT32 decoder expected projection map changed.");
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 2, "Full FLOAT32 subgraph count changed.");
      SubGraph decode = model.subgraphs(0);
      require(
          decode.tensorsLength() == FULL_DECODER_REWRITTEN_TENSOR_COUNT
              && decode.operatorsLength()
                  == FULL_DECODER_REWRITTEN_OPERATOR_COUNT
              && decode.outputsLength() == 1
              && decode.outputs(0) == REWRITTEN_LOGITS_TENSOR,
          "Full FLOAT32 decoder graph contract changed.");
      require(
          model.subgraphs(1).operatorsLength()
              == REWRITTEN_ENCODER_OPERATOR_COUNT,
          "Full FLOAT32 decoder encoder changed.");
      int floatProjectionCount = 0;
      int rankReshapeCount = 0;
      for (int operatorIndex = 0;
          operatorIndex < decode.operatorsLength();
          operatorIndex++) {
        Operator operator = decode.operators(operatorIndex);
        int builtin =
            model
                .operatorCodes(Math.toIntExact(operator.opcodeIndex()))
                .builtinCode();
        if (builtin == BuiltinOperator.FULLY_CONNECTED) {
          floatProjectionCount++;
          int weightTensor = operator.inputs(1);
          Tensor weight = decode.tensors(weightTensor);
          require(
              weight.type() == TensorType.FLOAT32
                  && weight.quantization() == null,
              "Full decoder projection is not FLOAT32 at operator "
                  + operatorIndex
                  + ".");
          Long offset = expectedOffsets.get(weightTensor);
          Long size = expectedSizes.get(weightTensor);
          require(
              offset != null
                  && size != null
                  && weight.buffer() > 0,
              "Full decoder projection tensor was not expected: "
                  + weightTensor
                  + ".");
          Buffer buffer =
              model.buffers(Math.toIntExact(weight.buffer()));
          require(
              buffer.dataLength() == 0
                  && readLongField(buffer, 6, 0L) == offset
                  && readLongField(buffer, 8, 0L) == size
                  && offset + size <= channel.size(),
              "Full decoder FLOAT32 buffer contract changed for tensor "
                  + weightTensor
                  + ".");
        }
        if (builtin == BuiltinOperator.RESHAPE
            && operator.outputsLength() == 1
            && operator.outputs(0) >= REWRITTEN_DECODER_TENSOR_COUNT) {
          rankReshapeCount++;
          require(
              operator.inputsLength() == 1
                  && Arrays.equals(
                      tensorShape(decode.tensors(operator.inputs(0))),
                      new int[] {20, 128, 128})
                  && Arrays.equals(
                      tensorShape(decode.tensors(operator.outputs(0))),
                      new int[] {1, 20, 128, 128})
                  && operatorIndex + 1 < decode.operatorsLength()
                  && decode.operators(operatorIndex + 1).inputs(0)
                      == operator.outputs(0),
              "Full decoder causal-mask rank barrier changed.");
        }
      }
      require(
          floatProjectionCount == FULL_DECODER_FLOAT_PROJECTION_COUNT,
          "Full decoder FLOAT32 projection count changed: "
              + floatProjectionCount
              + ".");
      require(
          rankReshapeCount == FULL_DECODER_MASK_RESHAPE_COUNT,
          "Full decoder mask rank-reshape count changed: "
              + rankReshapeCount
              + ".");
      SignatureDef signature = requireSignature(model, "decode");
      require(
          signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex()
                  == REWRITTEN_LOGITS_TENSOR,
          "Full FLOAT32 decoder signature changed.");
      requireNoImplicitAdjY(model, decode, decode.operatorsLength());
    }
  }

  private static ByteBuffer buildFloatProjectionTensorFragment(
      Tensor source, String label) {
    FlatBufferBuilder builder = new FlatBufferBuilder(1024);
    int[] shapeSignature =
        source.shapeSignatureLength() == 0
            ? null
            : tensorShapeSignature(source);
    int tensor =
        createTensor(
            builder,
            tensorShape(source),
            TensorType.FLOAT32,
            Math.toIntExact(source.buffer()),
            source.name() + "/voicememo_dequantized_float32_" + label,
            0,
            source.isVariable(),
            shapeSignature,
            readBooleanField(source, 20, false),
            fieldPresent(source, 20));
    builder.finish(tensor);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static String appendDequantizedProjectionWeights(
      String label,
      FileChannel output,
      long outputOffset,
      ByteBuffer quantizedWeights,
      QuantizationParameters quantization)
      throws Exception {
    require(
        quantizedWeights.remaining() == FIRST_BLOCK_K_WEIGHT_ELEMENTS,
        "First decoder " + label + " quantized weight byte count changed.");
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    ByteBuffer floats =
        ByteBuffer.allocateDirect(1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
    long writeOffset = outputOffset;
    for (int index = 0; index < FIRST_BLOCK_K_WEIGHT_ELEMENTS; index++) {
      int channel = index / FIRST_BLOCK_K_CHANNELS;
      long zeroPoint;
      if (quantization.zeroPointLength() == 0) {
        zeroPoint = 0L;
      } else if (quantization.zeroPointLength() == 1) {
        zeroPoint = quantization.zeroPoint(0);
      } else {
        zeroPoint = quantization.zeroPoint(channel);
      }
      float dequantized =
          (quantizedWeights.get(index) - zeroPoint)
              * quantization.scale(channel);
      if (floats.remaining() < Float.BYTES) {
        floats.flip();
        digest.update(floats.duplicate());
        int bytes = floats.remaining();
        writeFully(output, floats, writeOffset);
        writeOffset += bytes;
        floats.clear();
      }
      floats.putFloat(dequantized);
    }
    floats.flip();
    digest.update(floats.duplicate());
    int finalBytes = floats.remaining();
    writeFully(output, floats, writeOffset);
    writeOffset += finalBytes;
    require(
        writeOffset
            == outputOffset + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES,
        "FLOAT32 " + label + " weight byte count is inconsistent.");
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void generateSoftmaxProbe(Path output) throws Exception {
    require(
        hasBuildPathSegment(output),
        "Softmax-probe output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    FlatBufferBuilder builder = new FlatBufferBuilder(4096);
    int inputTensor =
        createTensor(
            builder,
            new int[] {20, 128, 128},
            TensorType.FLOAT32,
            0,
            "input_0",
            0,
            false,
            null,
            true,
            true);
    int outputTensor =
        createTensor(
            builder,
            new int[] {20, 128, 128},
            TensorType.FLOAT32,
            0,
            "output_0",
            0,
            false,
            null,
            true,
            true);
    int tensors =
        SubGraph.createTensorsVector(
            builder, new int[] {inputTensor, outputTensor});
    int options = SoftmaxOptions.createSoftmaxOptions(builder, 1.0f);
    int operator =
        createOperator(
            builder,
            0,
            new int[] {0},
            new int[] {1},
            BuiltinOptions.SoftmaxOptions,
            options,
            new byte[0],
            (byte) 0,
            new boolean[0],
            new int[0]);
    int operators = SubGraph.createOperatorsVector(builder, new int[] {operator});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {1});
    int graphName = builder.createString("softmax_probe");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, graphInputs, graphOutputs, operators, graphName);

    int inputName = builder.createString("input_0");
    int outputName = builder.createString("output_0");
    int inputMap = TensorMap.createTensorMap(builder, inputName, 0);
    int outputMap = TensorMap.createTensorMap(builder, outputName, 1);
    int signatureInputs =
        SignatureDef.createInputsVector(builder, new int[] {inputMap});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {outputMap});
    int signatureKey = builder.createString("softmax");
    int signature =
        SignatureDef.createSignatureDef(
            builder, signatureInputs, signatureOutputs, signatureKey, 0);

    int opcode =
        OperatorCode.createOperatorCode(
            builder,
            (byte) BuiltinOperator.SOFTMAX,
            0,
            1,
            BuiltinOperator.SOFTMAX);
    int opcodes = Model.createOperatorCodesVector(builder, new int[] {opcode});
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {graph});
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int buffers = Model.createBuffersVector(builder, new int[] {emptyBuffer});
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});
    int description = builder.createString("VoiceMemo isolated softmax diagnostic");
    int model =
        Model.createModel(
            builder,
            3,
            opcodes,
            subgraphs,
            description,
            buffers,
            0,
            0,
            signatures);
    Model.finishModelBuffer(builder, model);
    Files.write(
        output,
        builder.sizedByteArray(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    verifySoftmaxProbe(output);
    System.out.println("output=" + output);
    System.out.println("bytes=" + Files.size(output));
    System.out.println("sha256=" + sha256(output));
  }

  private static void generateMaskAddProbe(Path output) throws Exception {
    require(
        hasBuildPathSegment(output),
        "Mask-ADD probe output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    FlatBufferBuilder builder = new FlatBufferBuilder(4096);
    int scoreTensor =
        createTensor(
            builder,
            new int[] {20, 128, 128},
            TensorType.FLOAT32,
            0,
            "scores",
            0,
            false,
            null,
            true,
            true);
    int maskTensor =
        createTensor(
            builder,
            new int[] {1, 1, 128, 128},
            TensorType.FLOAT32,
            0,
            "mask",
            0,
            false,
            null,
            true,
            true);
    int outputTensor =
        createTensor(
            builder,
            new int[] {1, 20, 128, 128},
            TensorType.FLOAT32,
            0,
            "masked_scores",
            0,
            false,
            null,
            true,
            true);
    int tensors =
        SubGraph.createTensorsVector(
            builder, new int[] {scoreTensor, maskTensor, outputTensor});
    int options = AddOptions.createAddOptions(builder, (byte) 0, false);
    int operator =
        createOperator(
            builder,
            0,
            new int[] {0, 1},
            new int[] {2},
            BuiltinOptions.AddOptions,
            options,
            new byte[0],
            (byte) 0,
            new boolean[0],
            new int[0]);
    int operators = SubGraph.createOperatorsVector(builder, new int[] {operator});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0, 1});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
    int graphName = builder.createString("mask_add_probe");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, graphInputs, graphOutputs, operators, graphName);

    int scoreName = builder.createString("scores");
    int maskName = builder.createString("mask");
    int outputName = builder.createString("masked_scores");
    int scoreMap = TensorMap.createTensorMap(builder, scoreName, 0);
    int maskMap = TensorMap.createTensorMap(builder, maskName, 1);
    int outputMap = TensorMap.createTensorMap(builder, outputName, 2);
    int signatureInputs =
        SignatureDef.createInputsVector(builder, new int[] {scoreMap, maskMap});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {outputMap});
    int signatureKey = builder.createString("mask_add");
    int signature =
        SignatureDef.createSignatureDef(
            builder, signatureInputs, signatureOutputs, signatureKey, 0);

    int opcode =
        OperatorCode.createOperatorCode(
            builder,
            (byte) BuiltinOperator.ADD,
            0,
            1,
            BuiltinOperator.ADD);
    int opcodes = Model.createOperatorCodesVector(builder, new int[] {opcode});
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {graph});
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int buffers = Model.createBuffersVector(builder, new int[] {emptyBuffer});
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});
    int description =
        builder.createString("VoiceMemo isolated causal-mask ADD diagnostic");
    int model =
        Model.createModel(
            builder,
            3,
            opcodes,
            subgraphs,
            description,
            buffers,
            0,
            0,
            signatures);
    Model.finishModelBuffer(builder, model);
    Files.write(
        output,
        builder.sizedByteArray(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    verifyMaskAddProbe(output);
    System.out.println("output=" + output);
    System.out.println("bytes=" + Files.size(output));
    System.out.println("sha256=" + sha256(output));
  }

  private static void verifyMaskAddProbe(Path output) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 1, "Mask-ADD probe subgraph count changed.");
      require(model.operatorCodesLength() == 1, "Mask-ADD probe opcode count changed.");
      require(
          model.operatorCodes(0).builtinCode() == BuiltinOperator.ADD,
          "Mask-ADD probe opcode changed.");
      SubGraph graph = model.subgraphs(0);
      require(
          graph.operatorsLength() == 1
              && graph.inputsLength() == 2
              && graph.outputsLength() == 1,
          "Mask-ADD probe graph contract changed.");
      require(
          Arrays.equals(tensorShape(graph.tensors(0)), new int[] {20, 128, 128})
              && Arrays.equals(
                  tensorShape(graph.tensors(1)), new int[] {1, 1, 128, 128})
              && Arrays.equals(
                  tensorShape(graph.tensors(2)), new int[] {1, 20, 128, 128}),
          "Mask-ADD probe tensor shape changed.");
      SignatureDef signature = requireSignature(model, "mask_add");
      require(
          signature.inputsLength() == 2
              && signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 2,
          "Mask-ADD probe signature changed.");
    }
  }

  private static void verifySoftmaxProbe(Path output) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 1, "Softmax probe subgraph count changed.");
      require(model.operatorCodesLength() == 1, "Softmax probe opcode count changed.");
      require(
          model.operatorCodes(0).builtinCode() == BuiltinOperator.SOFTMAX,
          "Softmax probe opcode changed.");
      SubGraph graph = model.subgraphs(0);
      require(graph.operatorsLength() == 1, "Softmax probe operator count changed.");
      require(
          Arrays.equals(tensorShape(graph.tensors(0)), new int[] {20, 128, 128})
              && Arrays.equals(tensorShape(graph.tensors(1)), new int[] {20, 128, 128}),
          "Softmax probe tensor shape changed.");
      SignatureDef signature = requireSignature(model, "softmax");
      require(
          signature.inputsLength() == 1
              && signature.inputs(0).tensorIndex() == 0
              && signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 1,
          "Softmax probe signature changed.");
    }
  }

  private static void generateScoreProbe(Path output) throws Exception {
    require(
        hasBuildPathSegment(output),
        "Score-probe output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    FlatBufferBuilder builder = new FlatBufferBuilder(4096);
    int queryTensor =
        createTensor(
            builder,
            new int[] {20, 128, 64},
            TensorType.FLOAT32,
            0,
            "query",
            0,
            false,
            null,
            true,
            true);
    int keyTensor =
        createTensor(
            builder,
            new int[] {20, 64, 128},
            TensorType.FLOAT32,
            0,
            "key",
            0,
            false,
            null,
            true,
            true);
    int scoreTensor =
        createTensor(
            builder,
            new int[] {20, 128, 128},
            TensorType.FLOAT32,
            0,
            "scores",
            0,
            false,
            null,
            true,
            true);
    int tensors =
        SubGraph.createTensorsVector(
            builder, new int[] {queryTensor, keyTensor, scoreTensor});
    int options =
        BatchMatMulOptions.createBatchMatMulOptions(
            builder, false, false, false);
    int operator =
        createOperator(
            builder,
            0,
            new int[] {0, 1},
            new int[] {2},
            BuiltinOptions.BatchMatMulOptions,
            options,
            new byte[0],
            (byte) 0,
            new boolean[0],
            new int[0]);
    int operators = SubGraph.createOperatorsVector(builder, new int[] {operator});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0, 1});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
    int graphName = builder.createString("score_probe");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, graphInputs, graphOutputs, operators, graphName);

    int queryName = builder.createString("query");
    int keyName = builder.createString("key");
    int scoreName = builder.createString("scores");
    int queryMap = TensorMap.createTensorMap(builder, queryName, 0);
    int keyMap = TensorMap.createTensorMap(builder, keyName, 1);
    int scoreMap = TensorMap.createTensorMap(builder, scoreName, 2);
    int signatureInputs =
        SignatureDef.createInputsVector(builder, new int[] {queryMap, keyMap});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {scoreMap});
    int signatureKey = builder.createString("score");
    int signature =
        SignatureDef.createSignatureDef(
            builder, signatureInputs, signatureOutputs, signatureKey, 0);

    int opcode =
        OperatorCode.createOperatorCode(
            builder,
            (byte) BuiltinOperator.BATCH_MATMUL,
            0,
            1,
            BuiltinOperator.BATCH_MATMUL);
    int opcodes = Model.createOperatorCodesVector(builder, new int[] {opcode});
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {graph});
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int buffers = Model.createBuffersVector(builder, new int[] {emptyBuffer});
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});
    int description = builder.createString("VoiceMemo isolated score BMM diagnostic");
    int model =
        Model.createModel(
            builder,
            3,
            opcodes,
            subgraphs,
            description,
            buffers,
            0,
            0,
            signatures);
    Model.finishModelBuffer(builder, model);
    Files.write(
        output,
        builder.sizedByteArray(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    verifyScoreProbe(output);
    System.out.println("output=" + output);
    System.out.println("bytes=" + Files.size(output));
    System.out.println("sha256=" + sha256(output));
  }

  private static void verifyScoreProbe(Path output) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 1, "Score probe subgraph count changed.");
      require(model.operatorCodesLength() == 1, "Score probe opcode count changed.");
      require(
          model.operatorCodes(0).builtinCode() == BuiltinOperator.BATCH_MATMUL,
          "Score probe opcode changed.");
      SubGraph graph = model.subgraphs(0);
      require(graph.operatorsLength() == 1, "Score probe operator count changed.");
      Operator operator = graph.operators(0);
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(
          options != null && !options.adjX() && !options.adjY(),
          "Score probe unexpectedly adjusts an input.");
      require(
          Arrays.equals(tensorShape(graph.tensors(0)), new int[] {20, 128, 64})
              && Arrays.equals(tensorShape(graph.tensors(1)), new int[] {20, 64, 128})
              && Arrays.equals(tensorShape(graph.tensors(2)), new int[] {20, 128, 128}),
          "Score probe tensor shape changed.");
      SignatureDef signature = requireSignature(model, "score");
      require(
          signature.inputsLength() == 2
              && signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 2,
          "Score probe signature changed.");
    }
  }

  private static void generateValueProbe(Path output) throws Exception {
    require(
        hasBuildPathSegment(output),
        "Value-probe output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    FlatBufferBuilder builder = new FlatBufferBuilder(4096);
    int probabilityTensor =
        createTensor(
            builder,
            new int[] {20, 128, 128},
            TensorType.FLOAT32,
            0,
            "probabilities",
            0,
            false,
            null,
            true,
            true);
    int valueTensor =
        createTensor(
            builder,
            new int[] {20, 128, 64},
            TensorType.FLOAT32,
            0,
            "values",
            0,
            false,
            null,
            true,
            true);
    int contextTensor =
        createTensor(
            builder,
            new int[] {20, 128, 64},
            TensorType.FLOAT32,
            0,
            "context",
            0,
            false,
            null,
            true,
            true);
    int tensors =
        SubGraph.createTensorsVector(
            builder,
            new int[] {probabilityTensor, valueTensor, contextTensor});
    int options =
        BatchMatMulOptions.createBatchMatMulOptions(
            builder, false, false, false);
    int operator =
        createOperator(
            builder,
            0,
            new int[] {0, 1},
            new int[] {2},
            BuiltinOptions.BatchMatMulOptions,
            options,
            new byte[0],
            (byte) 0,
            new boolean[0],
            new int[0]);
    int operators = SubGraph.createOperatorsVector(builder, new int[] {operator});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0, 1});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
    int graphName = builder.createString("value_probe");
    int graph =
        SubGraph.createSubGraph(
            builder, tensors, graphInputs, graphOutputs, operators, graphName);

    int probabilityName = builder.createString("probabilities");
    int valueName = builder.createString("values");
    int contextName = builder.createString("context");
    int probabilityMap =
        TensorMap.createTensorMap(builder, probabilityName, 0);
    int valueMap = TensorMap.createTensorMap(builder, valueName, 1);
    int contextMap = TensorMap.createTensorMap(builder, contextName, 2);
    int signatureInputs =
        SignatureDef.createInputsVector(
            builder, new int[] {probabilityMap, valueMap});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {contextMap});
    int signatureKey = builder.createString("value_matmul");
    int signature =
        SignatureDef.createSignatureDef(
            builder, signatureInputs, signatureOutputs, signatureKey, 0);

    int opcode =
        OperatorCode.createOperatorCode(
            builder,
            (byte) BuiltinOperator.BATCH_MATMUL,
            0,
            1,
            BuiltinOperator.BATCH_MATMUL);
    int opcodes = Model.createOperatorCodesVector(builder, new int[] {opcode});
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {graph});
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int buffers = Model.createBuffersVector(builder, new int[] {emptyBuffer});
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});
    int description =
        builder.createString("VoiceMemo isolated attention-value BMM diagnostic");
    int model =
        Model.createModel(
            builder,
            3,
            opcodes,
            subgraphs,
            description,
            buffers,
            0,
            0,
            signatures);
    Model.finishModelBuffer(builder, model);
    Files.write(
        output,
        builder.sizedByteArray(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    verifyValueProbe(output);
    System.out.println("output=" + output);
    System.out.println("bytes=" + Files.size(output));
    System.out.println("sha256=" + sha256(output));
  }

  private static void verifyValueProbe(Path output) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 1, "Value probe subgraph count changed.");
      require(model.operatorCodesLength() == 1, "Value probe opcode count changed.");
      require(
          model.operatorCodes(0).builtinCode() == BuiltinOperator.BATCH_MATMUL,
          "Value probe opcode changed.");
      SubGraph graph = model.subgraphs(0);
      require(graph.operatorsLength() == 1, "Value probe operator count changed.");
      Operator operator = graph.operators(0);
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(
          options != null && !options.adjX() && !options.adjY(),
          "Value probe unexpectedly adjusts an input.");
      require(
          Arrays.equals(tensorShape(graph.tensors(0)), new int[] {20, 128, 128})
              && Arrays.equals(
                  tensorShape(graph.tensors(1)), new int[] {20, 128, 64})
              && Arrays.equals(
                  tensorShape(graph.tensors(2)), new int[] {20, 128, 64}),
          "Value probe tensor shape changed.");
      SignatureDef signature = requireSignature(model, "value_matmul");
      require(
          signature.inputsLength() == 2
              && signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 2,
          "Value probe signature changed.");
    }
  }

  private static void generateKProjectionProbe(Path source, Path output)
      throws Exception {
    require(Files.isRegularFile(source), "Rewritten production model is missing.");
    require(
        Files.size(source) == REWRITTEN_PRODUCTION_SIZE
            && REWRITTEN_PRODUCTION_SHA256.equals(sha256(source)),
        "K-projection source is not the pinned rewritten production model.");
    require(
        hasBuildPathSegment(output),
        "K-projection probe output must stay under a build directory.");
    Files.createDirectories(output.getParent());

    try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
      ByteBuffer sourceBytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model sourceModel = parseModel(sourceBytes);
      SubGraph decode = sourceModel.subgraphs(0);
      Operator projection = decode.operators(20);
      OperatorCode sourceCode =
          sourceModel.operatorCodes(Math.toIntExact(projection.opcodeIndex()));
      require(
          sourceCode.builtinCode() == BuiltinOperator.FULLY_CONNECTED,
          "K-projection operator changed.");
      require(
          projection.inputsLength() == 3
              && projection.inputs(0) == 133
              && projection.inputs(1) == 73
              && projection.inputs(2) == -1
              && projection.outputsLength() == 1
              && projection.outputs(0) == 134,
          "K-projection tensor contract changed.");
      Tensor hiddenSource = decode.tensors(133);
      Tensor weightSource = decode.tensors(73);
      Tensor keySource = decode.tensors(134);
      require(
          Arrays.equals(tensorShape(hiddenSource), DECODER_HIDDEN_SHAPE)
              && Arrays.equals(tensorShape(weightSource), new int[] {1280, 1280})
              && Arrays.equals(tensorShape(keySource), DECODER_HIDDEN_SHAPE),
          "K-projection tensor shape changed.");
      Buffer sourceWeightBuffer =
          sourceModel.buffers(Math.toIntExact(weightSource.buffer()));
      long externalOffset = readLongField(sourceWeightBuffer, 6, 0L);
      long externalSize = readLongField(sourceWeightBuffer, 8, 0L);
      require(
          sourceWeightBuffer.dataLength() > 0
              || (
                  externalOffset > 0
                      && externalSize > 0
                      && externalOffset + externalSize <= channel.size()),
          "K-projection weight storage is invalid.");

      FlatBufferBuilder builder = new FlatBufferBuilder(4 * 1024 * 1024);
      int hiddenTensor = cloneTensorWithBuffer(builder, hiddenSource, 0);
      int weightTensor = cloneTensorWithBuffer(builder, weightSource, 1);
      int keyTensor = cloneTensorWithBuffer(builder, keySource, 0);
      int tensors =
          SubGraph.createTensorsVector(
              builder, new int[] {hiddenTensor, weightTensor, keyTensor});
      int options = cloneBuiltinOptions(builder, projection, false);
      byte[] customOptions = new byte[projection.customOptionsLength()];
      for (int index = 0; index < customOptions.length; index++) {
        customOptions[index] = (byte) projection.customOptions(index);
      }
      boolean[] mutating = new boolean[projection.mutatingVariableInputsLength()];
      for (int index = 0; index < mutating.length; index++) {
        mutating[index] = projection.mutatingVariableInputs(index);
      }
      int operator =
          createOperator(
              builder,
              0,
              new int[] {0, 1, -1},
              new int[] {2},
              projection.builtinOptionsType(),
              options,
              customOptions,
              projection.customOptionsFormat(),
              mutating,
              intermediateVector(projection));
      int operators = SubGraph.createOperatorsVector(builder, new int[] {operator});
      int graphInputs = SubGraph.createInputsVector(builder, new int[] {0});
      int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
      int graphName = builder.createString("k_projection_probe");
      int graph =
          SubGraph.createSubGraph(
              builder, tensors, graphInputs, graphOutputs, operators, graphName);

      int hiddenName = builder.createString("hidden");
      int keyName = builder.createString("key");
      int hiddenMap = TensorMap.createTensorMap(builder, hiddenName, 0);
      int keyMap = TensorMap.createTensorMap(builder, keyName, 2);
      int signatureInputs =
          SignatureDef.createInputsVector(builder, new int[] {hiddenMap});
      int signatureOutputs =
          SignatureDef.createOutputsVector(builder, new int[] {keyMap});
      int signatureKey = builder.createString("k_projection");
      int signature =
          SignatureDef.createSignatureDef(
              builder, signatureInputs, signatureOutputs, signatureKey, 0);

      int opcode =
          OperatorCode.createOperatorCode(
              builder,
              (byte) BuiltinOperator.FULLY_CONNECTED,
              0,
              sourceCode.version(),
              BuiltinOperator.FULLY_CONNECTED);
      int opcodes = Model.createOperatorCodesVector(builder, new int[] {opcode});
      int subgraphs = Model.createSubgraphsVector(builder, new int[] {graph});
      int emptyBuffer = Buffer.createBuffer(builder, 0);
      ByteBuffer weightBytes;
      if (sourceWeightBuffer.dataLength() > 0) {
        weightBytes = sourceWeightBuffer.dataAsByteBuffer().duplicate();
      } else {
        ByteBuffer external = sourceBytes.duplicate();
        external.position(Math.toIntExact(externalOffset));
        external.limit(Math.toIntExact(externalOffset + externalSize));
        weightBytes = external.slice();
      }
      int weightData =
          Buffer.createDataVector(builder, weightBytes);
      int weightBuffer = Buffer.createBuffer(builder, weightData);
      int buffers =
          Model.createBuffersVector(builder, new int[] {emptyBuffer, weightBuffer});
      int signatures =
          Model.createSignatureDefsVector(builder, new int[] {signature});
      int description =
          builder.createString("VoiceMemo isolated INT8 K-projection diagnostic");
      int model =
          Model.createModel(
              builder,
              3,
              opcodes,
              subgraphs,
              description,
              buffers,
              0,
              0,
              signatures);
      Model.finishModelBuffer(builder, model);
      Files.write(
          output,
          builder.sizedByteArray(),
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING,
          StandardOpenOption.WRITE);
    }
    verifyKProjectionProbe(output);
    System.out.println("output=" + output);
    System.out.println("bytes=" + Files.size(output));
    System.out.println("sha256=" + sha256(output));
  }

  private static void verifyKProjectionProbe(Path output) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 1, "K-projection probe subgraph count changed.");
      require(
          model.operatorCodesLength() == 1
              && model.operatorCodes(0).builtinCode()
                  == BuiltinOperator.FULLY_CONNECTED,
          "K-projection probe opcode changed.");
      SubGraph graph = model.subgraphs(0);
      require(
          graph.operatorsLength() == 1
              && graph.inputsLength() == 1
              && graph.outputsLength() == 1,
          "K-projection probe graph contract changed.");
      require(
          Arrays.equals(tensorShape(graph.tensors(0)), DECODER_HIDDEN_SHAPE)
              && Arrays.equals(tensorShape(graph.tensors(1)), new int[] {1280, 1280})
              && Arrays.equals(tensorShape(graph.tensors(2)), DECODER_HIDDEN_SHAPE),
          "K-projection probe tensor shape changed.");
      require(
          model.buffersLength() == 2 && model.buffers(1).dataLength() > 0,
          "K-projection probe weight buffer changed.");
      SignatureDef signature = requireSignature(model, "k_projection");
      require(
          signature.inputsLength() == 1
              && signature.inputs(0).tensorIndex() == 0
              && signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == 2,
          "K-projection probe signature changed.");
    }
  }

  private static void generateKProjectionArithmeticProbes(
      Path source, Path outputDirectory) throws Exception {
    require(
        Files.isRegularFile(source)
            && Files.size(source) == K_PROJECTION_PROBE_SIZE
            && K_PROJECTION_PROBE_SHA256.equals(sha256(source)),
        "Arithmetic probes require the pinned isolated K-projection model.");
    require(
        hasBuildPathSegment(outputDirectory),
        "Arithmetic probes must stay under a build directory.");
    Files.createDirectories(outputDirectory);

    List<String> manifest = new ArrayList<>();
    manifest.add(
        "name\tmodel_sha256\tmodel_bytes\tweight_type\tscale_count"
            + "\tzero_point_count\tscales_sha256\tsums_sha256");
    try (FileChannel channel = FileChannel.open(source, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model sourceModel = parseModel(bytes);
      SubGraph graph = sourceModel.subgraphs(0);
      Operator projection = graph.operators(0);
      OperatorCode sourceCode =
          sourceModel.operatorCodes(Math.toIntExact(projection.opcodeIndex()));
      Tensor weight = graph.tensors(1);
      QuantizationParameters quantization = weight.quantization();
      require(
          weight.type() == TensorType.INT8
              && Arrays.equals(
                  tensorShape(weight),
                  new int[] {FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS})
              && quantization != null
              && quantization.detailsType() == QuantizationDetails.NONE
              && quantization.quantizedDimension() == 0
              && quantization.scaleLength() == FIRST_BLOCK_K_CHANNELS
              && (quantization.zeroPointLength() == 0
                  || quantization.zeroPointLength() == 1
                  || quantization.zeroPointLength() == FIRST_BLOCK_K_CHANNELS),
          "Pinned K-projection quantization contract changed.");
      ByteBuffer weightBuffer =
          sourceModel
              .buffers(Math.toIntExact(weight.buffer()))
              .dataAsByteBuffer()
              .duplicate();
      require(
          weightBuffer.remaining() == FIRST_BLOCK_K_WEIGHT_ELEMENTS,
          "Pinned K-projection weight size changed.");
      byte[] original = new byte[FIRST_BLOCK_K_WEIGHT_ELEMENTS];
      weightBuffer.get(original);
      float[] originalScales = new float[FIRST_BLOCK_K_CHANNELS];
      long[] originalZeroPoints =
          new long[quantization.zeroPointLength()];
      for (int index = 0; index < FIRST_BLOCK_K_CHANNELS; index++) {
        originalScales[index] = quantization.scale(index);
      }
      for (int index = 0; index < originalZeroPoints.length; index++) {
        originalZeroPoints[index] = quantization.zeroPoint(index);
      }
      float[] dequantized =
          dequantizeProjectionWeights(
              original, originalScales, originalZeroPoints);
      float maximumAbsoluteWeight = 0.0f;
      for (float value : dequantized) {
        maximumAbsoluteWeight =
            Math.max(maximumAbsoluteWeight, Math.abs(value));
      }
      require(maximumAbsoluteWeight > 0.0f, "K-projection weights are all zero.");
      float perTensorScale = maximumAbsoluteWeight / 127.0f;
      byte[] requantized =
          quantizeProjectionWeights(dequantized, perTensorScale);
      float[] uniformPerChannelScales =
          new float[FIRST_BLOCK_K_CHANNELS];
      Arrays.fill(uniformPerChannelScales, perTensorScale);
      long[] explicitZeroPoints = new long[FIRST_BLOCK_K_CHANNELS];

      byte[] diagonal = new byte[FIRST_BLOCK_K_WEIGHT_ELEMENTS];
      float[] diagonalScales = new float[FIRST_BLOCK_K_CHANNELS];
      for (int channelIndex = 0;
          channelIndex < FIRST_BLOCK_K_CHANNELS;
          channelIndex++) {
        diagonal[
            channelIndex * FIRST_BLOCK_K_CHANNELS + channelIndex] = 127;
        diagonalScales[channelIndex] =
            0.001f + channelIndex * 0.000001f;
      }

      byte[] accumulation = new byte[FIRST_BLOCK_K_WEIGHT_ELEMENTS];
      float[] accumulationScales = new float[FIRST_BLOCK_K_CHANNELS];
      Arrays.fill(accumulationScales, 0.01f);
      int[] accumulationWidths =
          new int[] {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1280};
      for (int channelIndex = 0;
          channelIndex < FIRST_BLOCK_K_CHANNELS;
          channelIndex++) {
        int width =
            accumulationWidths[channelIndex % accumulationWidths.length];
        Arrays.fill(
            accumulation,
            channelIndex * FIRST_BLOCK_K_CHANNELS,
            channelIndex * FIRST_BLOCK_K_CHANNELS + width,
            (byte) 1);
      }

      List<FcArithmeticVariant> variants =
          List.of(
              new FcArithmeticVariant(
                  "original-per-channel",
                  TensorType.INT8,
                  original,
                  originalScales,
                  originalZeroPoints),
              new FcArithmeticVariant(
                  "original-explicit-zero-points",
                  TensorType.INT8,
                  original,
                  originalScales,
                  explicitZeroPoints),
              new FcArithmeticVariant(
                  "requantized-per-tensor",
                  TensorType.INT8,
                  requantized,
                  new float[] {perTensorScale},
                  new long[] {0}),
              new FcArithmeticVariant(
                  "requantized-uniform-per-channel",
                  TensorType.INT8,
                  requantized,
                  uniformPerChannelScales,
                  explicitZeroPoints),
              new FcArithmeticVariant(
                  "synthetic-diagonal-scale-ramp",
                  TensorType.INT8,
                  diagonal,
                  diagonalScales,
                  explicitZeroPoints),
              new FcArithmeticVariant(
                  "synthetic-accumulation-ladder",
                  TensorType.INT8,
                  accumulation,
                  accumulationScales,
                  explicitZeroPoints),
              new FcArithmeticVariant(
                  "float32-original-control",
                  TensorType.FLOAT32,
                  floatsToLittleEndianBytes(dequantized),
                  new float[0],
                  new long[0]));

      for (FcArithmeticVariant variant : variants) {
        Path modelOutput =
            outputDirectory.resolve(variant.name() + ".tflite");
        writeKProjectionArithmeticProbe(
            projection, sourceCode.version(), variant, modelOutput);
        verifyKProjectionArithmeticProbe(modelOutput, variant);

        String scalesHash = "-";
        String sumsHash = "-";
        if (variant.weightType() == TensorType.INT8) {
          float[] analysisScales =
              expandScales(variant.scales());
          int[] sums = quantizedRowSums(variant.weightData());
          Path scalesOutput =
              outputDirectory.resolve(variant.name() + ".scales.f32");
          Path sumsOutput =
              outputDirectory.resolve(variant.name() + ".sums.i32");
          writeLittleEndianFloats(scalesOutput, analysisScales);
          writeLittleEndianInts(sumsOutput, sums);
          scalesHash = sha256(scalesOutput);
          sumsHash = sha256(sumsOutput);
        }
        manifest.add(
            variant.name()
                + "\t"
                + sha256(modelOutput)
                + "\t"
                + Files.size(modelOutput)
                + "\t"
                + variant.weightType()
                + "\t"
                + variant.scales().length
                + "\t"
                + variant.zeroPoints().length
                + "\t"
                + scalesHash
                + "\t"
                + sumsHash);
      }
    }
    Path manifestOutput = outputDirectory.resolve("matrix-manifest.tsv");
    Files.write(
        manifestOutput,
        manifest,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
    System.out.println("output_directory=" + outputDirectory);
    System.out.println("variants=" + (manifest.size() - 1));
    System.out.println("manifest_sha256=" + sha256(manifestOutput));
  }

  private static void writeKProjectionArithmeticProbe(
      Operator sourceProjection,
      int fullyConnectedVersion,
      FcArithmeticVariant variant,
      Path output) throws Exception {
    FlatBufferBuilder builder = new FlatBufferBuilder(
        variant.weightData().length + 2 * 1024 * 1024);
    int weightData =
        Buffer.createDataVector(builder, ByteBuffer.wrap(variant.weightData()));
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int constantWeightBuffer = Buffer.createBuffer(builder, weightData);
    int buffers =
        Model.createBuffersVector(
            builder, new int[] {emptyBuffer, constantWeightBuffer});

    int weightQuantization = 0;
    if (variant.weightType() == TensorType.INT8) {
      int scales =
          QuantizationParameters.createScaleVector(builder, variant.scales());
      int zeroPoints =
          variant.zeroPoints().length == 0
              ? 0
              : QuantizationParameters.createZeroPointVector(
                  builder, variant.zeroPoints());
      weightQuantization =
          QuantizationParameters.createQuantizationParameters(
              builder,
              0,
              0,
              scales,
              zeroPoints,
              QuantizationDetails.NONE,
              0,
              0);
    }
    int hiddenTensor =
        createTensor(
            builder,
            DECODER_HIDDEN_SHAPE,
            TensorType.FLOAT32,
            0,
            "hidden",
            0,
            false,
            null,
            true,
            true);
    int weightTensor =
        createTensor(
            builder,
            new int[] {FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS},
            variant.weightType(),
            1,
            "weight/" + variant.name(),
            weightQuantization,
            false,
            null,
            true,
            true);
    int outputTensor =
        createTensor(
            builder,
            DECODER_HIDDEN_SHAPE,
            TensorType.FLOAT32,
            0,
            "key",
            0,
            false,
            null,
            true,
            true);
    int tensors =
        SubGraph.createTensorsVector(
            builder, new int[] {hiddenTensor, weightTensor, outputTensor});
    int options = cloneBuiltinOptions(builder, sourceProjection, false);
    int projection =
        createOperator(
            builder,
            0,
            new int[] {0, 1, -1},
            new int[] {2},
            sourceProjection.builtinOptionsType(),
            options,
            new byte[0],
            sourceProjection.customOptionsFormat(),
            new boolean[0],
            new int[0]);
    int operators =
        SubGraph.createOperatorsVector(builder, new int[] {projection});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
    int graph =
        SubGraph.createSubGraph(
            builder,
            tensors,
            graphInputs,
            graphOutputs,
            operators,
            builder.createString("k_projection_" + variant.name()));

    int signatureInput =
        TensorMap.createTensorMap(builder, builder.createString("hidden"), 0);
    int signatureOutput =
        TensorMap.createTensorMap(builder, builder.createString("key"), 2);
    int signature =
        SignatureDef.createSignatureDef(
            builder,
            SignatureDef.createInputsVector(
                builder, new int[] {signatureInput}),
            SignatureDef.createOutputsVector(
                builder, new int[] {signatureOutput}),
            builder.createString("k_projection"),
            0);
    int opcode =
        OperatorCode.createOperatorCode(
            builder,
            (byte) BuiltinOperator.FULLY_CONNECTED,
            0,
            fullyConnectedVersion,
            BuiltinOperator.FULLY_CONNECTED);
    int model =
        Model.createModel(
            builder,
            3,
            Model.createOperatorCodesVector(builder, new int[] {opcode}),
            Model.createSubgraphsVector(builder, new int[] {graph}),
            builder.createString(
                "VoiceMemo INT8 FC arithmetic probe " + variant.name()),
            buffers,
            0,
            0,
            Model.createSignatureDefsVector(
                builder, new int[] {signature}));
    Model.finishModelBuffer(builder, model);
    Files.write(
        output,
        builder.sizedByteArray(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
  }

  private static void verifyKProjectionArithmeticProbe(
      Path output, FcArithmeticVariant variant) throws Exception {
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      SubGraph graph = model.subgraphs(0);
      require(
          model.subgraphsLength() == 1
              && model.operatorCodesLength() == 1
              && model.operatorCodes(0).builtinCode()
                  == BuiltinOperator.FULLY_CONNECTED
              && graph.operatorsLength() == 1,
          "Arithmetic probe operator contract changed.");
      Tensor weight = graph.tensors(1);
      require(
          weight.type() == variant.weightType()
              && Arrays.equals(
                  tensorShape(weight),
                  new int[] {FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS})
              && model.buffers(
                      Math.toIntExact(weight.buffer()))
                  .dataLength()
                  == variant.weightData().length,
          "Arithmetic probe weight contract changed.");
      QuantizationParameters quantization = weight.quantization();
      if (variant.weightType() == TensorType.INT8) {
        require(
            quantization != null
                && quantization.scaleLength() == variant.scales().length
                && quantization.zeroPointLength()
                    == variant.zeroPoints().length
                && quantization.quantizedDimension() == 0,
            "Arithmetic probe quantization contract changed.");
      } else {
        require(
            quantization == null || quantization.scaleLength() == 0,
            "FLOAT32 control unexpectedly has active quantization.");
      }
      requireSignature(model, "k_projection");
    }
  }

  private static float[] dequantizeProjectionWeights(
      byte[] weights, float[] scales, long[] zeroPoints) {
    float[] result = new float[weights.length];
    for (int row = 0; row < FIRST_BLOCK_K_CHANNELS; row++) {
      long zeroPoint =
          zeroPoints.length == 0
              ? 0
              : zeroPoints[Math.min(row, zeroPoints.length - 1)];
      float scale = scales[Math.min(row, scales.length - 1)];
      int offset = row * FIRST_BLOCK_K_CHANNELS;
      for (int column = 0; column < FIRST_BLOCK_K_CHANNELS; column++) {
        result[offset + column] =
            (weights[offset + column] - zeroPoint) * scale;
      }
    }
    return result;
  }

  private static byte[] quantizeProjectionWeights(
      float[] weights, float scale) {
    byte[] result = new byte[weights.length];
    for (int index = 0; index < weights.length; index++) {
      int quantized = Math.round(weights[index] / scale);
      result[index] = (byte) Math.max(-127, Math.min(127, quantized));
    }
    return result;
  }

  private static byte[] floatsToLittleEndianBytes(float[] values) {
    ByteBuffer result =
        ByteBuffer.allocate(values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    result.asFloatBuffer().put(values);
    return result.array();
  }

  private static float[] expandScales(float[] scales) {
    float[] result = new float[FIRST_BLOCK_K_CHANNELS];
    for (int index = 0; index < result.length; index++) {
      result[index] = scales[Math.min(index, scales.length - 1)];
    }
    return result;
  }

  private static int[] quantizedRowSums(byte[] weights) {
    int[] result = new int[FIRST_BLOCK_K_CHANNELS];
    for (int row = 0; row < FIRST_BLOCK_K_CHANNELS; row++) {
      int offset = row * FIRST_BLOCK_K_CHANNELS;
      int sum = 0;
      for (int column = 0; column < FIRST_BLOCK_K_CHANNELS; column++) {
        sum += weights[offset + column];
      }
      result[row] = sum;
    }
    return result;
  }

  private static void writeLittleEndianFloats(
      Path output, float[] values) throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Float.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    bytes.asFloatBuffer().put(values);
    Files.write(
        output,
        bytes.array(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
  }

  private static void writeLittleEndianInts(
      Path output, int[] values) throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate(values.length * Integer.BYTES)
            .order(ByteOrder.LITTLE_ENDIAN);
    bytes.asIntBuffer().put(values);
    Files.write(
        output,
        bytes.array(),
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING,
        StandardOpenOption.WRITE);
  }

  private static PrefixPatchLayout requireRewrittenPrefixContract(
      Model model, DecoderPrefixStage stage) {
    require(model.subgraphsLength() == 2, "Rewritten model subgraph count changed.");
    SubGraph decode = model.subgraphs(0);
    SubGraph encode = model.subgraphs(1);
    require(
        decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT,
        "Rewritten decoder tensor count changed.");
    require(
        decode.operatorsLength() == REWRITTEN_DECODER_OPERATOR_COUNT,
        "Rewritten decoder operator count changed.");
    require(
        encode.operatorsLength() == REWRITTEN_ENCODER_OPERATOR_COUNT,
        "Rewritten encoder operator count changed.");
    require(
        decode.outputsLength() == 1
            && decode.outputs(0) == REWRITTEN_LOGITS_TENSOR,
        "Rewritten decoder graph output changed.");
    require(
        stage.operatorCount() >= 1
            && stage.operatorCount() < decode.operatorsLength(),
        "Decoder-prefix operator count is invalid.");
    Operator producer = decode.operators(stage.operatorCount() - 1);
    require(
        producer.outputsLength() >= 1
            && containsOperatorOutput(producer, stage.outputTensor()),
        "Stage "
            + stage.id()
            + " boundary tensor is not produced by its final prefix operator.");
    Tensor boundary = decode.tensors(stage.outputTensor());
    require(
        Arrays.equals(tensorShape(boundary), stage.expectedShape()),
        "Stage " + stage.id() + " boundary shape changed: " + shape(boundary));
    requireNoImplicitAdjY(model, decode, stage.operatorCount());

    SignatureDef decodeSignature = requireSignature(model, "decode");
    require(
        decodeSignature.outputsLength() == 1,
        "Decode signature output count changed.");
    TensorMap signatureOutput = decodeSignature.outputs(0);
    require(
        signatureOutput.tensorIndex() == REWRITTEN_LOGITS_TENSOR,
        "Decode signature output tensor changed.");
    int operatorVector =
        tableVectorStart(decode, 10, "decode operator vector");
    int outputVector =
        tableVectorStart(decode, 8, "decode output vector");
    require(
        decode.getByteBuffer().getInt(operatorVector)
            == REWRITTEN_DECODER_OPERATOR_COUNT,
        "Decode operator vector length changed.");
    require(
        decode.getByteBuffer().getInt(outputVector) == 1,
        "Decode output vector length changed.");
    require(
        decode.getByteBuffer().getInt(outputVector + Integer.BYTES)
            == REWRITTEN_LOGITS_TENSOR,
        "Decode output vector tensor changed.");
    int signatureOutputField =
        tableFieldPosition(
            signatureOutput, 6, "decode signature output tensor");
    require(
        signatureOutput.getByteBuffer().getInt(signatureOutputField)
            == REWRITTEN_LOGITS_TENSOR,
        "Decode signature output field changed.");
    return new PrefixPatchLayout(
        operatorVector,
        outputVector + Integer.BYTES,
        signatureOutputField);
  }

  private static void verifyDecoderPrefix(
      Path output, DecoderPrefixStage stage) throws Exception {
    require(
        Files.size(output) == REWRITTEN_PRODUCTION_SIZE,
        "Generated decoder-prefix model size changed.");
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 2, "Prefix subgraph count changed.");
      SubGraph decode = model.subgraphs(0);
      require(
          decode.operatorsLength() == stage.operatorCount(),
          "Generated prefix operator count changed.");
      require(
          decode.outputsLength() == 1
              && decode.outputs(0) == stage.outputTensor(),
          "Generated prefix graph output changed.");
      SignatureDef signature = requireSignature(model, "decode");
      require(
          signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == stage.outputTensor(),
          "Generated prefix signature output changed.");
      requireNoImplicitAdjY(model, decode, decode.operatorsLength());
      Tensor boundary = decode.tensors(stage.outputTensor());
      require(
          Arrays.equals(tensorShape(boundary), stage.expectedShape()),
          "Generated prefix boundary shape changed.");
      require(
          model.subgraphs(1).operatorsLength()
              == REWRITTEN_ENCODER_OPERATOR_COUNT,
          "Generated prefix encoder graph changed.");
    }
  }

  private static void verifyFloatKDecoderPrefix(
      Path output,
      DecoderPrefixStage stage,
      long expectedWeightOffset,
      long expectedSize)
      throws Exception {
    require(
        Files.size(output) == expectedSize,
        "Generated FLOAT32 K-prefix model size changed.");
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(model.subgraphsLength() == 2, "FLOAT32 K-prefix subgraph count changed.");
      SubGraph decode = model.subgraphs(0);
      require(
          decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT
              && decode.operatorsLength() == stage.operatorCount(),
          "FLOAT32 K-prefix decoder graph contract changed.");
      require(
          decode.outputsLength() == 1
              && decode.outputs(0) == stage.outputTensor(),
          "FLOAT32 K-prefix graph output changed.");
      SignatureDef signature = requireSignature(model, "decode");
      require(
          signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == stage.outputTensor(),
          "FLOAT32 K-prefix signature output changed.");
      Tensor weight = decode.tensors(FIRST_BLOCK_K_WEIGHT_TENSOR);
      require(
          weight.type() == TensorType.FLOAT32
              && weight.quantization() == null
              && weight.buffer() == FIRST_BLOCK_K_WEIGHT_BUFFER
              && Arrays.equals(
                  tensorShape(weight),
                  new int[] {FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS}),
          "FLOAT32 K-prefix weight tensor contract changed.");
      Buffer weightBuffer = model.buffers(FIRST_BLOCK_K_WEIGHT_BUFFER);
      require(
          weightBuffer.dataLength() == 0
              && readLongField(weightBuffer, 6, 0L) == expectedWeightOffset
              && readLongField(weightBuffer, 8, 0L)
                  == FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES
              && expectedWeightOffset + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES
                  == channel.size(),
          "FLOAT32 K-prefix external weight buffer changed.");
      require(
          countTensorBufferReferences(decode, FIRST_BLOCK_K_WEIGHT_BUFFER) == 1,
          "FLOAT32 K-prefix weight buffer is unexpectedly shared.");
      Operator projection = decode.operators(FIRST_BLOCK_K_OPERATOR);
      require(
          projection.inputsLength() == 3
              && projection.inputs(0) == FIRST_BLOCK_K_INPUT_TENSOR
              && projection.inputs(1) == FIRST_BLOCK_K_WEIGHT_TENSOR
              && projection.inputs(2) == -1
              && projection.outputsLength() == 1
              && projection.outputs(0) == FIRST_BLOCK_K_OUTPUT_TENSOR,
          "FLOAT32 K-prefix projection contract changed.");
      requireNoImplicitAdjY(model, decode, decode.operatorsLength());
      require(
          model.subgraphs(1).operatorsLength()
              == REWRITTEN_ENCODER_OPERATOR_COUNT,
          "FLOAT32 K-prefix encoder graph changed.");
    }
  }

  private static void verifyFloatSelfAttentionDecoderPrefix(
      Path output,
      DecoderPrefixStage stage,
      long[] expectedWeightOffsets,
      long expectedSize)
      throws Exception {
    require(
        expectedWeightOffsets.length
            == FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length,
        "FLOAT32 self-attention verification offsets changed.");
    require(
        Files.size(output) == expectedSize,
        "Generated FLOAT32 self-attention prefix size changed.");
    try (FileChannel channel = FileChannel.open(output, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel
              .map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes);
      require(
          model.subgraphsLength() == 2,
          "FLOAT32 self-attention prefix subgraph count changed.");
      SubGraph decode = model.subgraphs(0);
      require(
          decode.tensorsLength() == REWRITTEN_DECODER_TENSOR_COUNT
              && decode.operatorsLength() == stage.operatorCount(),
          "FLOAT32 self-attention decoder graph contract changed.");
      require(
          decode.outputsLength() == 1
              && decode.outputs(0) == stage.outputTensor(),
          "FLOAT32 self-attention graph output changed.");
      SignatureDef signature = requireSignature(model, "decode");
      require(
          signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex() == stage.outputTensor(),
          "FLOAT32 self-attention signature output changed.");
      for (int index = 0;
          index < FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS.length;
          index++) {
        FloatProjectionSpec spec =
            FIRST_BLOCK_SELF_ATTENTION_PROJECTIONS[index];
        Tensor weight = decode.tensors(spec.weightTensor());
        require(
            weight.type() == TensorType.FLOAT32
                && weight.quantization() == null
                && weight.buffer() == spec.weightBuffer()
                && Arrays.equals(
                    tensorShape(weight),
                    new int[] {
                      FIRST_BLOCK_K_CHANNELS, FIRST_BLOCK_K_CHANNELS
                    }),
            "FLOAT32 "
                + spec.label()
                + " weight tensor contract changed.");
        Buffer weightBuffer = model.buffers(spec.weightBuffer());
        require(
            weightBuffer.dataLength() == 0
                && readLongField(weightBuffer, 6, 0L)
                    == expectedWeightOffsets[index]
                && readLongField(weightBuffer, 8, 0L)
                    == FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES
                && expectedWeightOffsets[index]
                        + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES
                    <= channel.size(),
            "FLOAT32 "
                + spec.label()
                + " external weight buffer changed.");
        require(
            countTensorBufferReferences(decode, spec.weightBuffer()) == 1,
            "FLOAT32 "
                + spec.label()
                + " weight buffer is unexpectedly shared.");
        if (spec.operatorIndex() < stage.operatorCount()) {
          Operator projection = decode.operators(spec.operatorIndex());
          require(
              projection.inputsLength() == 3
                  && projection.inputs(0) == spec.inputTensor()
                  && projection.inputs(1) == spec.weightTensor()
                  && projection.inputs(2) == spec.biasTensor()
                  && projection.outputsLength() == 1
                  && projection.outputs(0) == spec.outputTensor(),
              "FLOAT32 "
                  + spec.label()
                  + " projection contract changed.");
        }
      }
      long finalWeightEnd =
          expectedWeightOffsets[expectedWeightOffsets.length - 1]
              + FIRST_BLOCK_FLOAT_K_WEIGHT_BYTES;
      require(
          finalWeightEnd == channel.size(),
          "FLOAT32 self-attention final weight does not end at EOF.");
      requireNoImplicitAdjY(model, decode, decode.operatorsLength());
      require(
          model.subgraphs(1).operatorsLength()
              == REWRITTEN_ENCODER_OPERATOR_COUNT,
          "FLOAT32 self-attention encoder graph changed.");
    }
  }

  private static int countTensorBufferReferences(
      SubGraph graph, int bufferIndex) {
    int count = 0;
    for (int index = 0; index < graph.tensorsLength(); index++) {
      if (graph.tensors(index).buffer() == bufferIndex) count++;
    }
    return count;
  }

  private static int countModelTensorBufferReferences(
      Model model, int bufferIndex) {
    int count = 0;
    for (int graphIndex = 0;
        graphIndex < model.subgraphsLength();
        graphIndex++) {
      SubGraph graph = model.subgraphs(graphIndex);
      count += countTensorBufferReferences(graph, bufferIndex);
    }
    return count;
  }

  private static void requireNoImplicitAdjY(
      Model model, SubGraph graph, int operatorCount) {
    for (int index = 0; index < operatorCount; index++) {
      Operator operator = graph.operators(index);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(operator.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.BATCH_MATMUL) continue;
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(
          options != null && !options.adjY(),
          "Decoder prefix still contains adj_y=true at operator " + index + ".");
    }
  }

  private static SignatureDef requireSignature(Model model, String key) {
    for (int index = 0; index < model.signatureDefsLength(); index++) {
      SignatureDef signature = model.signatureDefs(index);
      // The cached schema artifact calls current signature_key methodName.
      if (key.equals(signature.methodName())) return signature;
    }
    throw new ToolFailure("Model signature is missing: " + key + ".");
  }

  private static boolean containsOperatorOutput(
      Operator operator, int tensorIndex) {
    for (int index = 0; index < operator.outputsLength(); index++) {
      if (operator.outputs(index) == tensorIndex) return true;
    }
    return false;
  }

  private static DecoderPrefixStage requireDecoderPrefixStage(String id) {
    for (DecoderPrefixStage stage : DECODER_PREFIX_STAGES) {
      if (stage.id().equals(id)) return stage;
    }
    throw new ToolFailure("Unsupported decoder-prefix stage: " + id + ".");
  }

  private static boolean isDecoderBoundaryShape(Tensor tensor) {
    if (tensor.shapeLength() != 3 || tensor.shape(0) != 1) return false;
    return tensor.shape(1) == 128
        && (tensor.shape(2) == 1280 || tensor.shape(2) == 51_866);
  }

  private static void inspectGraph(Model model, SubGraph graph, int graphIndex) {
    Map<String, Integer> optionCounts = new LinkedHashMap<>();
    int batchMatMulCount = 0;
    int adjYCount = 0;
    int transposeOpcode = -1;
    for (int index = 0; index < model.operatorCodesLength(); index++) {
      if (model.operatorCodes(index).builtinCode() == BuiltinOperator.TRANSPOSE) {
        transposeOpcode = index;
      }
    }
    int permutationTensor = findPermutationTensor(model, graph);
    int permutationBuffer = findPermutationBuffer(model);
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator operator = graph.operators(index);
      int optionType = Byte.toUnsignedInt(operator.builtinOptionsType());
      String optionName =
          optionType < BuiltinOptions.names.length
              ? BuiltinOptions.name(optionType)
              : "BuiltinOptions#" + optionType;
      optionCounts.merge(optionName, 1, Integer::sum);
      OperatorCode code =
          model.operatorCodes(Math.toIntExact(operator.opcodeIndex()));
      if (code.builtinCode() != BuiltinOperator.BATCH_MATMUL) continue;
      batchMatMulCount++;
      BatchMatMulOptions options =
          (BatchMatMulOptions)
              operator.builtinOptions(new BatchMatMulOptions());
      require(options != null, "BATCH_MATMUL " + index + " has no options.");
      if (!options.adjY()) continue;
      adjYCount++;
      require(operator.inputsLength() == 2, "BATCH_MATMUL input count changed.");
      Tensor left = graph.tensors(operator.inputs(0));
      Tensor right = graph.tensors(operator.inputs(1));
      System.out.println(
          "graph="
              + graphIndex
              + " op="
              + index
              + " adj_x="
              + options.adjX()
              + " adj_y="
              + options.adjY()
              + " left="
              + shape(left)
              + " right="
              + shape(right)
              + " output="
              + shape(graph.tensors(operator.outputs(0))));
    }
    System.out.println(
        "graph="
            + graphIndex
            + " name="
            + graph.name()
            + " tensors="
            + graph.tensorsLength()
            + " operators="
            + graph.operatorsLength()
            + " batch_matmul="
            + batchMatMulCount
            + " adj_y_true="
            + adjYCount
            + " transpose_opcode="
            + transposeOpcode
            + " perm_0_2_1_tensor="
            + permutationTensor
            + " perm_0_2_1_buffer="
            + permutationBuffer);
    System.out.println("graph=" + graphIndex + " builtin_options=" + optionCounts);
    System.out.println(
        "graph="
            + graphIndex
            + " newer_fields="
            + countNewerFields(graph)
            + " newer_field_offsets="
            + newerFieldCounts(graph)
            + " tensor_newer_field_offsets="
            + tensorNewerFieldCounts(graph)
            + " subgraph_debug_field="
            + fieldPresent(graph, 14));
  }

  private static int findPermutationTensor(Model model, SubGraph graph) {
    for (int tensorIndex = 0;
        tensorIndex < graph.tensorsLength();
        tensorIndex++) {
      Tensor tensor = graph.tensors(tensorIndex);
      if (tensor.type() != TensorType.INT32
          || tensor.shapeLength() != 1
          || tensor.shape(0) != 3) {
        continue;
      }
      Buffer buffer = model.buffers(Math.toIntExact(tensor.buffer()));
      if (buffer == null || buffer.dataLength() != 3 * Integer.BYTES) continue;
      ByteBuffer data = buffer.dataAsByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
      if (data.getInt() == 0 && data.getInt() == 2 && data.getInt() == 1) {
        return tensorIndex;
      }
    }
    return -1;
  }

  private static int findPermutationBuffer(Model model) {
    for (int bufferIndex = 0; bufferIndex < model.buffersLength(); bufferIndex++) {
      Buffer buffer = model.buffers(bufferIndex);
      if (buffer == null || buffer.dataLength() != 3 * Integer.BYTES) continue;
      ByteBuffer data = buffer.dataAsByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
      if (data.getInt() == 0 && data.getInt() == 2 && data.getInt() == 1) {
        return bufferIndex;
      }
    }
    return -1;
  }

  private static int countNewerFields(SubGraph graph) {
    int count = fieldPresent(graph, 14) ? 1 : 0;
    for (int index = 0; index < graph.tensorsLength(); index++) {
      Tensor tensor = graph.tensors(index);
      if (fieldPresent(tensor, 22)) count++;
      if (fieldPresent(tensor, 24)) count++;
      if (tensor.sparsity() != null) count++;
      if (tensor.quantization() != null
          && tensor.quantization().detailsType() != 0) {
        count++;
      }
    }
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator operator = graph.operators(index);
      for (int offset = 24; offset <= 30; offset += 2) {
        if (fieldPresent(operator, offset)) count++;
      }
    }
    return count;
  }

  private static Map<Integer, Integer> newerFieldCounts(SubGraph graph) {
    Map<Integer, Integer> counts = new LinkedHashMap<>();
    for (int index = 0; index < graph.operatorsLength(); index++) {
      Operator operator = graph.operators(index);
      for (int offset = 24; offset <= 30; offset += 2) {
        if (fieldPresent(operator, offset)) counts.merge(offset, 1, Integer::sum);
      }
    }
    return counts;
  }

  private static Map<Integer, Integer> tensorNewerFieldCounts(SubGraph graph) {
    Map<Integer, Integer> counts = new LinkedHashMap<>();
    for (int index = 0; index < graph.tensorsLength(); index++) {
      Tensor tensor = graph.tensors(index);
      for (int offset = 22; offset <= 24; offset += 2) {
        if (fieldPresent(tensor, offset)) counts.merge(offset, 1, Integer::sum);
      }
      if (tensor.sparsity() != null) counts.merge(16, 1, Integer::sum);
      if (tensor.quantization() != null
          && tensor.quantization().detailsType() != 0) {
        counts.merge(13, 1, Integer::sum);
      }
    }
    return counts;
  }

  private static boolean fieldPresent(Table table, int vtableOffset) {
    return tableOffset(table, vtableOffset) != 0;
  }

  private static int tableOffset(Table table, int vtableOffset) {
    try {
      return (int) TABLE_OFFSET.invoke(table, vtableOffset);
    } catch (ReflectiveOperationException error) {
      throw new ToolFailure("Could not inspect FlatBuffer fields.", error);
    }
  }

  private static int tablePosition(Table table) {
    try {
      return TABLE_POSITION.getInt(table);
    } catch (IllegalAccessException error) {
      throw new ToolFailure("Could not inspect a FlatBuffer table.", error);
    }
  }

  private static boolean readBooleanField(
      Table table, int vtableOffset, boolean defaultValue) {
    int relative = tableOffset(table, vtableOffset);
    return relative == 0
        ? defaultValue
        : table.getByteBuffer().get(tablePosition(table) + relative) != 0;
  }

  private static byte readByteField(
      Table table, int vtableOffset, byte defaultValue) {
    int relative = tableOffset(table, vtableOffset);
    return relative == 0
        ? defaultValue
        : table.getByteBuffer().get(tablePosition(table) + relative);
  }

  private static int readIntField(
      Table table, int vtableOffset, int defaultValue) {
    int relative = tableOffset(table, vtableOffset);
    return relative == 0
        ? defaultValue
        : table.getByteBuffer().getInt(tablePosition(table) + relative);
  }

  private static long readLongField(
      Table table, int vtableOffset, long defaultValue) {
    int relative = tableOffset(table, vtableOffset);
    return relative == 0
        ? defaultValue
        : table.getByteBuffer().getLong(tablePosition(table) + relative);
  }

  private static int[] readIntVector(Table table, int vtableOffset) {
    int relative = tableOffset(table, vtableOffset);
    if (relative == 0) return new int[0];
    ByteBuffer bytes = table.getByteBuffer();
    int field = Math.addExact(tablePosition(table), relative);
    int vector = Math.addExact(field, bytes.getInt(field));
    int length = bytes.getInt(vector);
    require(length >= 0, "FlatBuffer INT32 vector has a negative length.");
    int[] result = new int[length];
    int data = Math.addExact(vector, Integer.BYTES);
    for (int index = 0; index < length; index++) {
      result[index] =
          bytes.getInt(Math.addExact(data, index * Integer.BYTES));
    }
    return result;
  }

  private static int createIntVector(
      FlatBufferBuilder builder, int[] values) {
    builder.startVector(Integer.BYTES, values.length, Integer.BYTES);
    for (int index = values.length - 1; index >= 0; index--) {
      builder.addInt(values[index]);
    }
    return builder.endVector();
  }

  private static int tableVectorStart(
      Table table, int vtableOffset, String label) {
    int relative = tableOffset(table, vtableOffset);
    require(relative != 0, label + " field is absent.");
    int field = Math.addExact(tablePosition(table), relative);
    int vector = Math.addExact(field, table.getByteBuffer().getInt(field));
    require(vector >= 0 && vector < table.getByteBuffer().limit(),
        label + " vector is invalid.");
    return vector;
  }

  private static int tableFieldPosition(
      Table table, int vtableOffset, String label) {
    int relative = tableOffset(table, vtableOffset);
    require(relative != 0, label + " field is absent.");
    return Math.addExact(tablePosition(table), relative);
  }

  private static void writeForwardOffset(
      FileChannel channel, int field, int target, String label)
      throws IOException {
    require(target > field, label + " target must point forward.");
    ByteBuffer value =
        ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    value.putInt(target - field).flip();
    writeFully(channel, value, field);
  }

  private static void writeInt(FileChannel channel, long offset, int value)
      throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeLong(FileChannel channel, long offset, long value)
      throws IOException {
    ByteBuffer bytes =
        ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putLong(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void writeFully(
      FileChannel channel, ByteBuffer source, long offset) throws IOException {
    long position = offset;
    while (source.hasRemaining()) {
      int written = channel.write(source, position);
      require(written > 0, "Could not write output at byte " + position + ".");
      position += written;
    }
  }

  private static long align(long value, int alignment) {
    return Math.addExact(value, alignment - 1L) & -alignment;
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

  private static String shape(Tensor tensor) {
    StringBuilder result = new StringBuilder("[");
    for (int index = 0; index < tensor.shapeLength(); index++) {
      if (index > 0) result.append(',');
      result.append(tensor.shape(index));
    }
    return result.append(']').toString();
  }

  private static Model parseModel(ByteBuffer bytes) {
    require(
        Model.ModelBufferHasIdentifier(bytes.duplicate()),
        "Model does not have the TFL3 identifier.");
    Model model = Model.getRootAsModel(bytes);
    require(model.version() == 3, "Model schema version is not 3.");
    return model;
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new ToolFailure(message);
  }

  private record DecoderPrefixStage(
      String id, int operatorCount, int outputTensor, int[] expectedShape) {}

  private record PrefixPatchLayout(
      int operatorVector, int graphOutputElement, int signatureOutputField) {}

  private record FloatProjectionSpec(
      String label,
      int operatorIndex,
      int inputTensor,
      int weightTensor,
      int biasTensor,
      int outputTensor,
      int weightBuffer) {}

  private record FcArithmeticVariant(
      String name,
      byte weightType,
      byte[] weightData,
      float[] scales,
      long[] zeroPoints) {}

  private record ProjectionSource(
      Tensor weight,
      Buffer weightBuffer,
      QuantizationParameters quantization,
      long quantizedWeightOffset,
      long quantizedWeightSize) {}

  private record FullDecoderProjection(
      int operatorIndex,
      int weightTensor,
      int weightBuffer,
      Tensor weight,
      Buffer buffer,
      QuantizationParameters quantization,
      long quantizedOffset,
      long quantizedSize,
      int outputChannels,
      int innerElements,
      long elementCount,
      String label) {}

  private record SourceContract(
      String label,
      long size,
      String sha256,
      int[] rewritesPerSubgraph) {
    int totalRewrites() {
      int result = 0;
      for (int count : rewritesPerSubgraph) result += count;
      return result;
    }
  }

  private static final class ToolFailure extends RuntimeException {
    ToolFailure(String message) {
      super(message);
    }

    ToolFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }
}
