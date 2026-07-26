package com.example.voicememo.tools;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import org.tensorflow.lite.schema.ActivationFunctionType;
import org.tensorflow.lite.schema.AddOptions;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.BuiltinOperator;
import org.tensorflow.lite.schema.BuiltinOptions;
import org.tensorflow.lite.schema.CustomOptionsFormat;
import org.tensorflow.lite.schema.FullyConnectedOptions;
import org.tensorflow.lite.schema.FullyConnectedOptionsWeightsFormat;
import org.tensorflow.lite.schema.Model;
import org.tensorflow.lite.schema.MulOptions;
import org.tensorflow.lite.schema.Operator;
import org.tensorflow.lite.schema.OperatorCode;
import org.tensorflow.lite.schema.QuantizationDetails;
import org.tensorflow.lite.schema.QuantizationParameters;
import org.tensorflow.lite.schema.ReducerOptions;
import org.tensorflow.lite.schema.SignatureDef;
import org.tensorflow.lite.schema.SubGraph;
import org.tensorflow.lite.schema.SubOptions;
import org.tensorflow.lite.schema.Tensor;
import org.tensorflow.lite.schema.TensorMap;
import org.tensorflow.lite.schema.TensorType;

/**
 * Produces decoder-split diagnostic models from the exact pinned Whisper
 * FlatBuffer. This is a developer-side tool; generated files never replace the
 * production asset and are intentionally written under the ignored build tree.
 */
public final class WhisperDecoderSplitTool {
  private static final long ORIGINAL_SIZE = 755_273_648L;
  private static final String ORIGINAL_SHA256 =
      "da3c91fcd149174cbb5abd3a5583ea95982c5e401c2d68cabac89117f5ce1a4c";

  private static final String CORE_FILE =
      "whisper_large_v3_turbo_30s_i4_decode_hidden.tflite";
  private static final String PROJECTION_FILE =
      "whisper_large_v3_turbo_active_row_projection_i8.tflite";
  private static final String PRE_LAYER_NORM_CORE_FILE =
      "whisper_large_v3_turbo_30s_i4_decode_pre_layer_norm.tflite";
  private static final String LAYER_NORM_PROJECTION_FILE =
      "whisper_large_v3_turbo_active_row_layer_norm_projection_i8.tflite";
  private static final String MANIFEST_FILE = "manifest.json";
  private static final String BOUNDARY_MANIFEST_SUFFIX = ".manifest.json";

  private static final int DECODE_SUBGRAPH = 0;
  private static final int ORIGINAL_DECODE_OPERATOR_COUNT = 321;
  private static final int CORE_DECODE_OPERATOR_COUNT = 320;
  private static final int PRE_LAYER_NORM_CORE_OPERATOR_COUNT = 310;
  private static final int LAYER_NORM_FIRST_OPERATOR_INDEX = 310;
  private static final int LAYER_NORM_TAIL_OPERATOR_COUNT = 11;
  private static final int PROJECTION_OPERATOR_INDEX = 320;
  private static final int PRE_LAYER_NORM_TENSOR_INDEX = 423;
  private static final int HIDDEN_TENSOR_INDEX = 433;
  private static final int BETA_TENSOR_INDEX = 101;
  private static final int GAMMA_TENSOR_INDEX = 102;
  private static final int WEIGHT_TENSOR_INDEX = 103;
  private static final int EPSILON_TENSOR_INDEX = 112;
  private static final int REDUCTION_AXIS_TENSOR_INDEX = 113;
  private static final int LOGITS_TENSOR_INDEX = 434;
  private static final int VOCABULARY_SIZE = 51_866;
  private static final int HIDDEN_WIDTH = 1_280;
  private static final int DECODER_LENGTH = 128;
  private static final int LAYER_NORM_PARAMETER_SIZE = HIDDEN_WIDTH * Float.BYTES;
  private static final float LAYER_NORM_EPSILON = 1.0e-5f;
  private static final long BETA_OFFSET = 54_517_168L;
  private static final long GAMMA_OFFSET = 54_522_288L;
  private static final long PROJECTION_WEIGHT_OFFSET = 54_527_408L;
  private static final String BETA_SHA256 =
      "f499d2848c5afb3d96ea541a781cecf7ebc2a31d75d6bf41dca7d3fd5cd94209";
  private static final String GAMMA_SHA256 =
      "0c46b5f643cd11d3ea0cbe4d0742617b4ac4af49129ac14371e8105ca4c3b6cd";
  private static final String EPSILON_SHA256 =
      "e6540c5b3269d9a6e3646dd482c2a7fe6b58e101e80007556892bb770099a68c";
  private static final int CURRENT_TENSOR_FIELD_COUNT = 11;
  private static final int TENSOR_HAS_RANK_FIELD = 8;
  private static final int TENSOR_HAS_RANK_VTABLE_OFFSET = 20;
  private static final long PROJECTION_WEIGHT_SIZE =
      (long) VOCABULARY_SIZE * HIDDEN_WIDTH;

  private static final int TAIL_INPUT = 0;
  private static final int TAIL_BETA = 1;
  private static final int TAIL_GAMMA = 2;
  private static final int TAIL_WEIGHT = 3;
  private static final int TAIL_EPSILON = 4;
  private static final int TAIL_AXIS = 5;
  private static final int TAIL_MEAN = 6;
  private static final int TAIL_SQUARED_DIFFERENCE = 7;
  private static final int TAIL_VARIANCE = 8;
  private static final int TAIL_VARIANCE_PLUS_EPSILON = 9;
  private static final int TAIL_RECIPROCAL_STDDEV = 10;
  private static final int TAIL_SCALED_GAMMA = 11;
  private static final int TAIL_SCALED_INPUT = 12;
  private static final int TAIL_SCALED_MEAN = 13;
  private static final int TAIL_CENTERING_TERM = 14;
  private static final int TAIL_NORMALIZED = 15;
  private static final int TAIL_LOGITS = 16;
  private static final int TAIL_TENSOR_COUNT = 17;
  private static final int TAIL_OPCODE_ADD = 0;
  private static final int TAIL_OPCODE_MEAN = 1;
  private static final int TAIL_OPCODE_SQUARED_DIFFERENCE = 2;
  private static final int TAIL_OPCODE_RSQRT = 3;
  private static final int TAIL_OPCODE_FULLY_CONNECTED = 4;
  private static final int TAIL_OPCODE_MUL = 5;
  private static final int TAIL_OPCODE_SUB = 6;

  private static final BoundaryStage[] BOUNDARY_STAGES = {
      new BoundaryStage(
          "after-position-add",
          "whisper_large_v3_turbo_decode_shared_after_position_add.tflite",
          2,
          115,
          "hidden_after_position_add",
          "gpu",
          "cpu",
          755_273_920L,
          "ee557aa9924a66f0b54f70c80759804b137e77dbaa9661d3b12bb5291615495d"),
      new BoundaryStage(
          "after-cross-attention",
          "whisper_large_v3_turbo_decode_shared_after_cross_attention.tflite",
          65,
          178,
          "hidden_after_cross_attention",
          "gpu",
          "cpu",
          755_273_928L,
          "bf71f075f3dffee2d7d233fffc0d77559e624941424a24933830e94386df3f5a"),
      new BoundaryStage(
          "after-1-block",
          "whisper_large_v3_turbo_decode_shared_after_1_block.tflite",
          79,
          192,
          "hidden_after_1_decoder_block",
          "gpu",
          "cpu",
          755_273_920L,
          "ed01a9e67f3a5991580bc7892ec9f747d63ed587bab02a6ef1b5c72366d4a6f3"),
      new BoundaryStage(
          "after-2-blocks",
          "whisper_large_v3_turbo_decode_shared_after_2_blocks.tflite",
          156,
          269,
          "hidden_after_2_decoder_blocks",
          "gpu",
          "cpu",
          755_273_920L,
          "f5926743061104e49dd12f532fe3d35a3025f791bc1ec75c4e2a0b7eef136337"),
      new BoundaryStage(
          "after-3-blocks",
          "whisper_large_v3_turbo_decode_shared_after_3_blocks.tflite",
          233,
          346,
          "hidden_after_3_decoder_blocks",
          "gpu",
          "cpu",
          755_273_920L,
          "9f4deb00a29115753cbdb0a663de08c34eec2079f9c15c5f98ebb314cbacb3ab"),
  };

  // Stable only for ORIGINAL_SHA256. Every patch validates its old value
  // before writing, and the finished file is parsed and hashed again.
  private static final Patch[] CORE_PATCHES = {
      new Patch(224L, LOGITS_TENSOR_INDEX, HIDDEN_TENSOR_INDEX,
          "decode signature output"),
      new Patch(328_796L, ORIGINAL_DECODE_OPERATOR_COUNT,
          CORE_DECODE_OPERATOR_COUNT, "decode operator count"),
      new Patch(344_428L, LOGITS_TENSOR_INDEX, HIDDEN_TENSOR_INDEX,
          "decode graph output"),
  };
  private static final Patch[] PRE_LAYER_NORM_CORE_PATCHES = {
      new Patch(224L, LOGITS_TENSOR_INDEX, PRE_LAYER_NORM_TENSOR_INDEX,
          "pre-layer-norm decode signature output"),
      new Patch(328_796L, ORIGINAL_DECODE_OPERATOR_COUNT,
          PRE_LAYER_NORM_CORE_OPERATOR_COUNT,
          "pre-layer-norm decode operator count"),
      new Patch(344_428L, LOGITS_TENSOR_INDEX, PRE_LAYER_NORM_TENSOR_INDEX,
          "pre-layer-norm decode graph output"),
  };

  // Filled from deterministic output after the generator was first validated.
  // Keeping these pinned makes a changed toolchain or accidental graph change
  // fail closed instead of silently creating a new diagnostic contract.
  private static final long EXPECTED_CORE_SIZE = ORIGINAL_SIZE;
  private static final String EXPECTED_CORE_SHA256 =
      "a73a5d6acdd2e9271b7bced77ba784bad2c9716f4946cdc9145e0d70d97e1d0b";
  private static final long EXPECTED_PROJECTION_SIZE = 67_011_640L;
  private static final String EXPECTED_PROJECTION_SHA256 =
      "220e0fa46e2bdcf3d0396762fb1bd39df169615458107ed2af28184c78ebbb35";
  private static final long EXPECTED_PRE_LAYER_NORM_CORE_SIZE = ORIGINAL_SIZE;
  private static final String EXPECTED_PRE_LAYER_NORM_CORE_SHA256 =
      "5006280fc6a6b2be0d029bd411c7dd2348de7afda9b7c757d2210397e14afe01";
  private static final long EXPECTED_LAYER_NORM_PROJECTION_SIZE = 67_023_504L;
  private static final String EXPECTED_LAYER_NORM_PROJECTION_SHA256 =
      "10ba23e2dc4e49e162d30f3e91d3b1c510f66d1e2cee63e5b7e4a3af37a4fe9a";

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

  private WhisperDecoderSplitTool() {}

  public static void main(String[] rawArguments) {
    try {
      Arguments arguments = Arguments.parse(rawArguments);
      switch (arguments.command()) {
        case "generate" -> generate(arguments);
        case "verify" -> verify(arguments);
        case "generate-boundary" -> generateBoundary(arguments);
        case "verify-boundary" -> verifyBoundary(arguments);
        case "self-test" -> selfTest();
        default -> throw new ToolFailure("Unsupported command: " + arguments.command());
      }
    } catch (ToolFailure error) {
      System.err.println("Whisper decoder split failed: " + error.getMessage());
      System.exit(1);
    } catch (Throwable error) {
      error.printStackTrace(System.err);
      System.exit(1);
    }
  }

  private static void generate(Arguments arguments) throws IOException {
    Path projectRoot = arguments.requiredProjectRoot();
    Path input = arguments.requiredInput();
    Path outputDirectory = arguments.requiredOutput();
    requireRegularFile(input, "Pinned Whisper model");
    Path projectBuild = requireProjectBuildOutput(projectRoot, outputDirectory);
    Files.createDirectories(outputDirectory);
    require(outputDirectory.toRealPath().startsWith(projectBuild),
        "Diagnostic output resolved outside the project build/ directory.");
    rejectProductionOverwrite(input, outputDirectory);

    SourceContract source = inspectPinnedSource(input);
    Path core = outputDirectory.resolve(CORE_FILE);
    Path projection = outputDirectory.resolve(PROJECTION_FILE);
    Path preLayerNormCore = outputDirectory.resolve(PRE_LAYER_NORM_CORE_FILE);
    Path layerNormProjection =
        outputDirectory.resolve(LAYER_NORM_PROJECTION_FILE);

    if (arguments.force() || !isValidCore(core)) {
      generateCore(input, core);
    }
    if (arguments.force() || !isValidProjection(projection)) {
      generateProjection(input, projection, source);
    }
    if (arguments.force() || !isValidPreLayerNormCore(preLayerNormCore)) {
      generatePreLayerNormCore(input, preLayerNormCore);
    }
    if (arguments.force()
        || !isValidLayerNormProjection(layerNormProjection)) {
      generateLayerNormProjection(input, layerNormProjection, source);
    }

    Artifact coreArtifact = verifyCore(core);
    Artifact projectionArtifact = verifyProjection(projection, source);
    Artifact preLayerNormCoreArtifact =
        verifyPreLayerNormCore(preLayerNormCore);
    Artifact layerNormProjectionArtifact =
        verifyLayerNormProjection(layerNormProjection, source);
    verifyPinnedGeneratedHashes(
        coreArtifact,
        projectionArtifact,
        preLayerNormCoreArtifact,
        layerNormProjectionArtifact);
    writeManifest(
        outputDirectory.resolve(MANIFEST_FILE),
        source,
        coreArtifact,
        projectionArtifact,
        preLayerNormCoreArtifact,
        layerNormProjectionArtifact);
    printSummary(
        input,
        outputDirectory,
        source,
        coreArtifact,
        projectionArtifact,
        preLayerNormCoreArtifact,
        layerNormProjectionArtifact);
  }

  private static void verify(Arguments arguments) throws IOException {
    Path input = arguments.requiredInput();
    Path outputDirectory = arguments.requiredOutput();
    SourceContract source = inspectPinnedSource(input);
    Artifact core = verifyCore(outputDirectory.resolve(CORE_FILE));
    Artifact projection =
        verifyProjection(outputDirectory.resolve(PROJECTION_FILE), source);
    Artifact preLayerNormCore =
        verifyPreLayerNormCore(outputDirectory.resolve(PRE_LAYER_NORM_CORE_FILE));
    Artifact layerNormProjection = verifyLayerNormProjection(
        outputDirectory.resolve(LAYER_NORM_PROJECTION_FILE),
        source);
    verifyPinnedGeneratedHashes(
        core,
        projection,
        preLayerNormCore,
        layerNormProjection);
    printSummary(
        input,
        outputDirectory,
        source,
        core,
        projection,
        preLayerNormCore,
        layerNormProjection);
  }

  private static void generateBoundary(Arguments arguments) throws IOException {
    Path projectRoot = arguments.requiredProjectRoot();
    Path input = arguments.requiredInput();
    Path outputDirectory = arguments.requiredOutput();
    BoundaryStage stage = arguments.requiredBoundaryStage();
    requireRegularFile(input, "Pinned Whisper model");
    Path projectBuild = requireProjectBuildOutput(projectRoot, outputDirectory);
    Files.createDirectories(outputDirectory);
    require(outputDirectory.toRealPath().startsWith(projectBuild),
        "Diagnostic output resolved outside the project build/ directory.");
    rejectProductionOverwrite(input, outputDirectory);

    SourceContract source = inspectPinnedSource(input);
    Path model = outputDirectory.resolve(stage.fileName());
    if (arguments.force() || !isValidBoundaryModel(model, input, stage)) {
      generateBoundaryModel(input, model, stage);
    }
    Artifact artifact = verifyBoundaryModel(model, input, stage);
    verifyPinnedBoundaryHash(artifact, stage);
    writeBoundaryManifest(
        outputDirectory.resolve(stage.fileName() + BOUNDARY_MANIFEST_SUFFIX),
        source,
        stage,
        artifact);
    printBoundarySummary(input, outputDirectory, stage, artifact);
  }

  private static void verifyBoundary(Arguments arguments) throws IOException {
    Path input = arguments.requiredInput();
    Path outputDirectory = arguments.requiredOutput();
    BoundaryStage stage = arguments.requiredBoundaryStage();
    Artifact artifact = verifyBoundaryModel(
        outputDirectory.resolve(stage.fileName()),
        input,
        stage);
    verifyPinnedBoundaryHash(artifact, stage);
    printBoundarySummary(input, outputDirectory, stage, artifact);
  }

  private static SourceContract inspectPinnedSource(Path input) throws IOException {
    requireRegularFile(input, "Pinned Whisper model");
    long size = Files.size(input);
    require(size == ORIGINAL_SIZE,
        "Pinned model size is " + size + "; expected " + ORIGINAL_SIZE + ".");
    String sha256 = sha256(input);
    require(ORIGINAL_SHA256.equals(sha256),
        "Pinned model SHA-256 is " + sha256 + "; expected " + ORIGINAL_SHA256 + ".");

    try (FileChannel channel = FileChannel.open(input, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes, "Pinned Whisper model");
      require(model.subgraphsLength() == 2,
          "Pinned model must contain decode and encode subgraphs.");
      SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
      require("decode".equals(decode.name()), "Subgraph 0 must be decode.");
      require(decode.operatorsLength() == ORIGINAL_DECODE_OPERATOR_COUNT,
          "Decode operator count changed: " + decode.operatorsLength() + ".");
      require(decode.outputsLength() == 1
              && decode.outputs(0) == LOGITS_TENSOR_INDEX,
          "Decode graph output is not tensor " + LOGITS_TENSOR_INDEX + ".");
      require(findSignatureOutput(model, "decode") == LOGITS_TENSOR_INDEX,
          "Decode signature output is not tensor " + LOGITS_TENSOR_INDEX + ".");

      Tensor hidden = decode.tensors(HIDDEN_TENSOR_INDEX);
      requireTensor(
          hidden,
          TensorType.FLOAT32,
          new int[] {1, DECODER_LENGTH, HIDDEN_WIDTH},
          "Decoder hidden tensor");
      require(hidden.buffer() == 0, "Decoder hidden tensor must be dynamic.");

      Tensor preLayerNorm = decode.tensors(PRE_LAYER_NORM_TENSOR_INDEX);
      requireTensor(
          preLayerNorm,
          TensorType.FLOAT32,
          new int[] {1, DECODER_LENGTH, HIDDEN_WIDTH},
          "Decoder pre-layer-norm tensor");
      require(preLayerNorm.buffer() == 0,
          "Decoder pre-layer-norm tensor must be dynamic.");
      requireLayerNormTail(model, decode);

      Operator projection = decode.operators(PROJECTION_OPERATOR_INDEX);
      requireProjectionOperator(model, decode, projection);

      Tensor weight = decode.tensors(WEIGHT_TENSOR_INDEX);
      requireTensor(
          weight,
          TensorType.INT8,
          new int[] {VOCABULARY_SIZE, HIDDEN_WIDTH},
          "Projection weight tensor");
      QuantizationParameters quantization = weight.quantization();
      require(quantization != null, "Projection weight quantization is missing.");
      require(quantization.scaleLength() == VOCABULARY_SIZE,
          "Projection weight scale count changed.");
      require(quantization.zeroPointLength() == VOCABULARY_SIZE,
          "Projection weight zero-point count changed.");
      require(quantization.quantizedDimension() == 0,
          "Projection weight quantized dimension must be 0.");
      require(quantization.detailsType() == QuantizationDetails.NONE,
          "Projection weight must use affine, not blockwise, quantization.");

      float[] scales = new float[VOCABULARY_SIZE];
      long[] zeroPoints = new long[VOCABULARY_SIZE];
      float minimumScale = Float.POSITIVE_INFINITY;
      float maximumScale = Float.NEGATIVE_INFINITY;
      for (int index = 0; index < VOCABULARY_SIZE; index++) {
        scales[index] = quantization.scale(index);
        zeroPoints[index] = quantization.zeroPoint(index);
        minimumScale = Math.min(minimumScale, scales[index]);
        maximumScale = Math.max(maximumScale, scales[index]);
        require(zeroPoints[index] == 0,
            "Projection weight zero point " + index + " is not zero.");
      }

      Buffer weightBuffer = model.buffers((int) weight.buffer());
      long weightOffset = tableLong(weightBuffer, 6);
      long weightSize = tableLong(weightBuffer, 8);
      require(weightBuffer.dataLength() == 0,
          "Pinned projection weight must use an external in-file buffer.");
      require(weightSize == PROJECTION_WEIGHT_SIZE,
          "Projection weight byte size changed: " + weightSize + ".");
      require(weightOffset > 1 && weightOffset + weightSize <= size,
          "Projection weight external buffer is outside the model file.");
      require(weightOffset == PROJECTION_WEIGHT_OFFSET,
          "Projection weight offset changed: " + weightOffset + ".");
      String weightSha256 = sha256Region(channel, weightOffset, weightSize);

      Tensor beta = decode.tensors(BETA_TENSOR_INDEX);
      requireTensor(
          beta,
          TensorType.FLOAT32,
          new int[] {HIDDEN_WIDTH},
          "Layer-norm beta tensor");
      ConstantRegion betaRegion = requireExternalConstant(
          model,
          beta,
          channel,
          size,
          BETA_OFFSET,
          LAYER_NORM_PARAMETER_SIZE,
          BETA_SHA256,
          "Layer-norm beta");

      Tensor gamma = decode.tensors(GAMMA_TENSOR_INDEX);
      requireTensor(
          gamma,
          TensorType.FLOAT32,
          new int[] {HIDDEN_WIDTH},
          "Layer-norm gamma tensor");
      ConstantRegion gammaRegion = requireExternalConstant(
          model,
          gamma,
          channel,
          size,
          GAMMA_OFFSET,
          LAYER_NORM_PARAMETER_SIZE,
          GAMMA_SHA256,
          "Layer-norm gamma");

      Tensor epsilon = decode.tensors(EPSILON_TENSOR_INDEX);
      requireTensor(
          epsilon,
          TensorType.FLOAT32,
          new int[] {},
          "Layer-norm epsilon tensor");
      ByteBuffer epsilonBytes =
          requireInlineConstant(model, epsilon, Float.BYTES, "Layer-norm epsilon");
      require(
          epsilonBytes.order(ByteOrder.LITTLE_ENDIAN).getInt(0)
              == Float.floatToIntBits(LAYER_NORM_EPSILON),
          "Layer-norm epsilon changed.");
      require(EPSILON_SHA256.equals(sha256(epsilonBytes.duplicate())),
          "Layer-norm epsilon bytes changed.");

      Tensor reductionAxis = decode.tensors(REDUCTION_AXIS_TENSOR_INDEX);
      requireTensor(
          reductionAxis,
          TensorType.INT32,
          new int[] {1},
          "Layer-norm reduction-axis tensor");
      ByteBuffer reductionAxisBytes = requireInlineConstant(
          model,
          reductionAxis,
          Integer.BYTES,
          "Layer-norm reduction axis");
      require(reductionAxisBytes.order(ByteOrder.LITTLE_ENDIAN).getInt(0) == 2,
          "Source layer-norm reduction axis must be decoder dimension 2.");

      for (Patch patch : CORE_PATCHES) {
        require(readLittleEndianInt(channel, patch.offset()) == patch.expected(),
            "Pinned model patch guard failed for " + patch.description() + ".");
      }

      return new SourceContract(
          size,
          sha256,
          weightOffset,
          weightSize,
          weightSha256,
          scales,
          zeroPoints,
          minimumScale,
          maximumScale,
          betaRegion,
          gammaRegion,
          Float.floatToIntBits(LAYER_NORM_EPSILON));
    }
  }

  private static void requireProjectionOperator(
      Model model,
      SubGraph graph,
      Operator projection) {
    OperatorCode code = model.operatorCodes((int) projection.opcodeIndex());
    require(code.builtinCode() == BuiltinOperator.FULLY_CONNECTED,
        "Projection op is not FULLY_CONNECTED.");
    require(code.version() == 5, "Projection FULLY_CONNECTED version is not 5.");
    require(projection.inputsLength() == 3
            && projection.inputs(0) == HIDDEN_TENSOR_INDEX
            && projection.inputs(1) == WEIGHT_TENSOR_INDEX
            && projection.inputs(2) == -1,
        "Projection op inputs changed.");
    require(projection.outputsLength() == 1
            && projection.outputs(0) == LOGITS_TENSOR_INDEX,
        "Projection op output changed.");
    FullyConnectedOptions options =
        (FullyConnectedOptions) projection.builtinOptions(
            new FullyConnectedOptions());
    require(options != null, "Projection FULLY_CONNECTED options are missing.");
    require(options.fusedActivationFunction() == ActivationFunctionType.NONE,
        "Projection activation must be NONE.");
    require(options.weightsFormat()
            == FullyConnectedOptionsWeightsFormat.DEFAULT,
        "Projection weight format must be DEFAULT.");
    require(options.keepNumDims(), "Projection must keep input dimensions.");
    require(!options.asymmetricQuantizeInputs(),
        "Projection asymmetric input quantization must be disabled.");

    Tensor output = graph.tensors(LOGITS_TENSOR_INDEX);
    requireTensor(
        output,
        TensorType.FLOAT32,
        new int[] {1, DECODER_LENGTH, VOCABULARY_SIZE},
        "Projection logits tensor");
  }

  private static void requireLayerNormTail(Model model, SubGraph graph) {
    int[] tensorIndices = {
        424, 425, 426, 427, 428, 429, 430, 431, 432, HIDDEN_TENSOR_INDEX,
    };
    int[][] tensorShapes = {
        {1, DECODER_LENGTH, 1},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
        {1, DECODER_LENGTH, 1},
        {1, DECODER_LENGTH, 1},
        {1, DECODER_LENGTH, 1},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
        {1, DECODER_LENGTH, HIDDEN_WIDTH},
    };
    for (int index = 0; index < tensorIndices.length; index++) {
      Tensor tensor = graph.tensors(tensorIndices[index]);
      requireTensor(
          tensor,
          TensorType.FLOAT32,
          tensorShapes[index],
          "Source layer-norm tensor " + tensorIndices[index]);
      require(tensor.buffer() == 0,
          "Source layer-norm tensor " + tensorIndices[index]
              + " must be dynamic.");
    }

    Operator boundary = requireOperator(
        model,
        graph,
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT - 1,
        BuiltinOperator.ADD,
        1,
        new int[] {409, 422},
        new int[] {PRE_LAYER_NORM_TENSOR_INDEX},
        "Pre-layer-norm residual ADD");
    requireAddOptions(boundary, "Pre-layer-norm residual ADD");

    Operator mean = requireOperator(
        model,
        graph,
        310,
        BuiltinOperator.MEAN,
        1,
        new int[] {423, REDUCTION_AXIS_TENSOR_INDEX},
        new int[] {424},
        "Layer-norm mean");
    requireReducerOptions(mean, "Layer-norm mean");
    requireNoOptions(
        requireOperator(
            model,
            graph,
            311,
            BuiltinOperator.SQUARED_DIFFERENCE,
            1,
            new int[] {423, 424},
            new int[] {425},
            "Layer-norm squared difference"),
        "Layer-norm squared difference");
    Operator variance = requireOperator(
        model,
        graph,
        312,
        BuiltinOperator.MEAN,
        1,
        new int[] {425, REDUCTION_AXIS_TENSOR_INDEX},
        new int[] {426},
        "Layer-norm variance");
    requireReducerOptions(variance, "Layer-norm variance");
    requireAddOptions(
        requireOperator(
            model,
            graph,
            313,
            BuiltinOperator.ADD,
            1,
            new int[] {426, EPSILON_TENSOR_INDEX},
            new int[] {427},
            "Layer-norm epsilon ADD"),
        "Layer-norm epsilon ADD");
    requireNoOptions(
        requireOperator(
            model,
            graph,
            314,
            BuiltinOperator.RSQRT,
            1,
            new int[] {427},
            new int[] {428},
            "Layer-norm reciprocal square root"),
        "Layer-norm reciprocal square root");
    requireMulOptions(
        requireOperator(
            model,
            graph,
            315,
            BuiltinOperator.MUL,
            1,
            new int[] {428, GAMMA_TENSOR_INDEX},
            new int[] {429},
            "Layer-norm scaled gamma"),
        "Layer-norm scaled gamma");
    requireMulOptions(
        requireOperator(
            model,
            graph,
            316,
            BuiltinOperator.MUL,
            1,
            new int[] {423, 429},
            new int[] {430},
            "Layer-norm scaled input"),
        "Layer-norm scaled input");
    requireMulOptions(
        requireOperator(
            model,
            graph,
            317,
            BuiltinOperator.MUL,
            1,
            new int[] {424, 429},
            new int[] {431},
            "Layer-norm scaled mean"),
        "Layer-norm scaled mean");
    requireSubOptions(
        requireOperator(
            model,
            graph,
            318,
            BuiltinOperator.SUB,
            1,
            new int[] {BETA_TENSOR_INDEX, 431},
            new int[] {432},
            "Layer-norm centering term"),
        "Layer-norm centering term");
    requireAddOptions(
        requireOperator(
            model,
            graph,
            319,
            BuiltinOperator.ADD,
            1,
            new int[] {430, 432},
            new int[] {HIDDEN_TENSOR_INDEX},
            "Layer-norm output ADD"),
        "Layer-norm output ADD");
  }

  private static Operator requireOperator(
      Model model,
      SubGraph graph,
      int operatorIndex,
      int builtinCode,
      int version,
      int[] inputs,
      int[] outputs,
      String label) {
    Operator operator = graph.operators(operatorIndex);
    require(operator != null, label + " is missing.");
    OperatorCode code = model.operatorCodes((int) operator.opcodeIndex());
    require(code != null
            && code.builtinCode() == builtinCode
            && code.version() == version,
        label + " opcode or version changed.");
    require(operatorVectorEquals(operator, true, inputs),
        label + " inputs changed.");
    require(operatorVectorEquals(operator, false, outputs),
        label + " outputs changed.");
    require(operator.customOptionsLength() == 0,
        label + " unexpectedly has custom options.");
    require(operator.intermediatesLength() == 0,
        label + " unexpectedly has intermediate tensors.");
    return operator;
  }

  private static boolean operatorVectorEquals(
      Operator operator,
      boolean input,
      int[] expected) {
    int length = input ? operator.inputsLength() : operator.outputsLength();
    if (length != expected.length) return false;
    for (int index = 0; index < expected.length; index++) {
      int actual = input ? operator.inputs(index) : operator.outputs(index);
      if (actual != expected[index]) return false;
    }
    return true;
  }

  private static void requireReducerOptions(Operator operator, String label) {
    require(operator.builtinOptionsType() == BuiltinOptions.ReducerOptions,
        label + " must use ReducerOptions.");
    ReducerOptions options =
        (ReducerOptions) operator.builtinOptions(new ReducerOptions());
    require(options != null && options.keepDims(),
        label + " must preserve reduced dimensions.");
  }

  private static void requireAddOptions(Operator operator, String label) {
    require(operator.builtinOptionsType() == BuiltinOptions.AddOptions,
        label + " must use AddOptions.");
    AddOptions options = (AddOptions) operator.builtinOptions(new AddOptions());
    require(options != null
            && options.fusedActivationFunction() == ActivationFunctionType.NONE
            && options.potScaleInt16(),
        label + " options changed.");
  }

  private static void requireMulOptions(Operator operator, String label) {
    require(operator.builtinOptionsType() == BuiltinOptions.MulOptions,
        label + " must use MulOptions.");
    MulOptions options = (MulOptions) operator.builtinOptions(new MulOptions());
    require(options != null
            && options.fusedActivationFunction() == ActivationFunctionType.NONE,
        label + " options changed.");
  }

  private static void requireSubOptions(Operator operator, String label) {
    require(operator.builtinOptionsType() == BuiltinOptions.SubOptions,
        label + " must use SubOptions.");
    SubOptions options = (SubOptions) operator.builtinOptions(new SubOptions());
    require(options != null
            && options.fusedActivationFunction() == ActivationFunctionType.NONE
            && options.potScaleInt16(),
        label + " options changed.");
  }

  private static void requireNoOptions(Operator operator, String label) {
    require(operator.builtinOptionsType() == BuiltinOptions.NONE,
        label + " must not have builtin options.");
  }

  private static ConstantRegion requireExternalConstant(
      Model model,
      Tensor tensor,
      FileChannel channel,
      long modelSize,
      long expectedOffset,
      long expectedSize,
      String expectedSha256,
      String label) throws IOException {
    Buffer buffer = model.buffers((int) tensor.buffer());
    require(buffer != null && buffer.dataLength() == 0,
        label + " must use an external in-file buffer.");
    long offset = tableLong(buffer, 6);
    long size = tableLong(buffer, 8);
    require(offset == expectedOffset,
        label + " offset changed: " + offset + ".");
    require(size == expectedSize,
        label + " size changed: " + size + ".");
    require(offset > 1 && offset + size <= modelSize,
        label + " is outside the model file.");
    String sha256 = sha256Region(channel, offset, size);
    require(expectedSha256.equals(sha256),
        label + " SHA-256 changed: " + sha256 + ".");
    return new ConstantRegion(offset, size, sha256);
  }

  private static ByteBuffer requireInlineConstant(
      Model model,
      Tensor tensor,
      int expectedSize,
      String label) {
    Buffer buffer = model.buffers((int) tensor.buffer());
    require(buffer != null && buffer.dataLength() == expectedSize,
        label + " inline byte size changed.");
    require(tableLong(buffer, 6) == 0 && tableLong(buffer, 8) == 0,
        label + " unexpectedly uses an external buffer.");
    ByteBuffer bytes = buffer.dataAsByteBuffer();
    require(bytes != null && bytes.remaining() == expectedSize,
        label + " inline bytes are unavailable.");
    return bytes.slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void generateCore(Path input, Path output) throws IOException {
    generatePatchedCore(
        input,
        output,
        CORE_FILE,
        CORE_PATCHES,
        EXPECTED_CORE_SIZE,
        EXPECTED_CORE_SHA256,
        CORE_DECODE_OPERATOR_COUNT,
        HIDDEN_TENSOR_INDEX,
        "post-layer-norm GPU core");
  }

  private static void generatePreLayerNormCore(Path input, Path output)
      throws IOException {
    generatePatchedCore(
        input,
        output,
        PRE_LAYER_NORM_CORE_FILE,
        PRE_LAYER_NORM_CORE_PATCHES,
        EXPECTED_PRE_LAYER_NORM_CORE_SIZE,
        EXPECTED_PRE_LAYER_NORM_CORE_SHA256,
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT,
        PRE_LAYER_NORM_TENSOR_INDEX,
        "pre-layer-norm GPU core");
  }

  private static void generatePatchedCore(
      Path input,
      Path output,
      String fileName,
      Patch[] patches,
      long expectedSize,
      String expectedSha256,
      int expectedOperatorCount,
      int expectedOutputTensor,
      String label) throws IOException {
    Path temporary = createSiblingTemporary(output);
    try {
      Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
      try (FileChannel channel =
          FileChannel.open(
              temporary,
              StandardOpenOption.READ,
              StandardOpenOption.WRITE)) {
        for (Patch patch : patches) {
          int actual = readLittleEndianInt(channel, patch.offset());
          require(actual == patch.expected(),
              label + " patch guard failed for " + patch.description()
                  + ": found " + actual + ", expected " + patch.expected() + ".");
          writeLittleEndianInt(channel, patch.offset(), patch.replacement());
        }
        channel.force(true);
      }
      Artifact candidate = new Artifact(
          fileName,
          Files.size(temporary),
          sha256(temporary));
      require(candidate.size() == expectedSize,
          "Generated " + label + " size changed: " + candidate.size() + ".");
      if (!expectedSha256.isEmpty()) {
        require(expectedSha256.equals(candidate.sha256()),
            "Generated " + label + " SHA-256 changed: "
                + candidate.sha256() + ".");
      }
      moveAtomically(temporary, output);
      // Parse only after publication. Mapping a file and then moving it is
      // unsupported on Windows; the exact pre-move hash is the structural
      // identity guard for the candidate.
      verifyCoreContract(
          output,
          expectedOperatorCount,
          expectedOutputTensor,
          patches,
          label);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void generateBoundaryModel(
      Path input,
      Path output,
      BoundaryStage stage) throws IOException {
    Path temporary = createSiblingTemporary(output);
    try {
      Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
      try (FileChannel channel =
          FileChannel.open(
              temporary,
              StandardOpenOption.READ,
              StandardOpenOption.WRITE)) {
        ByteBuffer sourceBytes =
            channel.map(FileChannel.MapMode.READ_ONLY, 0, ORIGINAL_SIZE)
                .order(ByteOrder.LITTLE_ENDIAN);
        BoundaryLayout layout = inspectBoundaryLayout(
            sourceBytes,
            stage,
            "Pinned boundary source");
        ByteBuffer signatureFragment = buildBoundarySignature(stage);
        int fragmentLength = signatureFragment.remaining();
        long fragmentOffset = channel.size();
        require((fragmentOffset & 3L) == 0L,
            "Boundary signature fragment is not four-byte aligned.");
        int fragmentRoot = signatureFragment.getInt(signatureFragment.position());
        require(fragmentRoot > 0 && fragmentRoot < fragmentLength,
            "Boundary signature fragment root is invalid.");
        writeFully(channel, signatureFragment, fragmentOffset);

        writeLittleEndianInt(
            channel,
            layout.decodeOperatorVector(),
            stage.prefixOperatorCount());
        writeLittleEndianInt(
            channel,
            layout.decodeOutputElement(),
            stage.boundaryTensor());
        writeLittleEndianInt(
            channel,
            layout.decodeSignatureOutputField(),
            stage.boundaryTensor());

        writeLittleEndianInt(
            channel,
            layout.tailTensorVector(),
            layout.decodeTensorTargets().length);
        for (int index = 0;
            index < layout.decodeTensorTargets().length;
            index++) {
          int element = layout.tailTensorVector() + Integer.BYTES
              + index * Integer.BYTES;
          writeForwardOffset(
              channel,
              element,
              layout.decodeTensorTargets()[index],
              "tail tensor " + index);
        }

        writeLittleEndianInt(
            channel,
            layout.tailOperatorVector(),
            layout.decodeTailOperatorTargets().length);
        for (int index = 0;
            index < layout.decodeTailOperatorTargets().length;
            index++) {
          int element = layout.tailOperatorVector() + Integer.BYTES
              + index * Integer.BYTES;
          writeForwardOffset(
              channel,
              element,
              layout.decodeTailOperatorTargets()[index],
              "tail operator " + index);
        }

        writeLittleEndianInt(channel, layout.tailInputVector(), 3);
        writeLittleEndianInt(
            channel,
            layout.tailInputVector() + Integer.BYTES,
            1);
        writeLittleEndianInt(
            channel,
            layout.tailInputVector() + 2 * Integer.BYTES,
            stage.boundaryTensor());
        writeLittleEndianInt(
            channel,
            layout.tailInputVector() + 3 * Integer.BYTES,
            0);
        writeForwardOffset(
            channel,
            layout.tailInputField(),
            layout.tailInputVector(),
            "tail input vector");

        writeLittleEndianInt(channel, layout.tailOutputVector(), 1);
        writeLittleEndianInt(
            channel,
            layout.tailOutputVector() + Integer.BYTES,
            PRE_LAYER_NORM_TENSOR_INDEX);
        writeForwardOffset(
            channel,
            layout.tailOutputField(),
            layout.tailOutputVector(),
            "tail output vector");

        long signatureTarget = fragmentOffset + fragmentRoot;
        require(signatureTarget <= Integer.MAX_VALUE,
            "Boundary signature target exceeds FlatBuffer offset range.");
        writeForwardOffset(
            channel,
            layout.secondSignatureVectorElement(),
            Math.toIntExact(signatureTarget),
            "tail signature");
        channel.force(true);
      }

      moveAtomically(temporary, output);
      verifyBoundaryModel(output, input, stage);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static BoundaryLayout inspectBoundaryLayout(
      ByteBuffer bytes,
      BoundaryStage stage,
      String label) {
    Model model = parseModel(bytes, label);
    require(model.subgraphsLength() == 2,
        label + " must contain two subgraphs.");
    require(model.signatureDefsLength() == 2,
        label + " must contain two signatures.");
    SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
    SubGraph reusableTail = model.subgraphs(1);
    require(decode.operatorsLength() == ORIGINAL_DECODE_OPERATOR_COUNT,
        label + " decode operator count changed.");
    require(decode.tensorsLength() == 476,
        label + " decode tensor count changed.");
    require(reusableTail.operatorsLength() >= ORIGINAL_DECODE_OPERATOR_COUNT,
        label + " encoder operator-vector capacity is too small.");
    require(reusableTail.tensorsLength() >= decode.tensorsLength(),
        label + " encoder tensor-vector capacity is too small.");

    int decodeOperatorVector = tableVectorStart(
        decode,
        10,
        label + " decode operators");
    int decodeOutputVector = tableVectorStart(
        decode,
        8,
        label + " decode outputs");
    require(bytes.getInt(decodeOutputVector) == 1,
        label + " decode output vector length changed.");
    SignatureDef decodeSignature = findSignature(model, "decode");
    TensorMap decodeSignatureOutput = decodeSignature.outputs(0);
    int decodeSignatureOutputField = tableFieldPosition(
        decodeSignatureOutput,
        6,
        label + " decode signature output");

    int tailTensorVector = tableVectorStart(
        reusableTail,
        4,
        label + " reusable tensor vector");
    int tailOperatorVector = tableVectorStart(
        reusableTail,
        10,
        label + " reusable operator vector");
    int tailInputField = tableFieldPosition(
        reusableTail,
        6,
        label + " reusable input field");
    int tailOutputField = tableFieldPosition(
        reusableTail,
        8,
        label + " reusable output field");
    int tailOperatorCapacity = bytes.getInt(tailOperatorVector);
    int tailOperatorCount =
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT - stage.prefixOperatorCount();
    require(tailOperatorCount > 0,
        "Boundary stage must leave at least one decoder operator for the tail.");
    require(tailOperatorCapacity >= tailOperatorCount + 16,
        "Reusable operator vector has no safe scratch capacity.");

    int tailInputVector = tailOperatorVector + Integer.BYTES
        + (tailOperatorCount + 8) * Integer.BYTES;
    int tailOutputVector = tailInputVector + 4 * Integer.BYTES;
    int tailOperatorStorageEnd = tailOperatorVector + Integer.BYTES
        + tailOperatorCapacity * Integer.BYTES;
    require(tailOutputVector + 2 * Integer.BYTES <= tailOperatorStorageEnd,
        "Reusable operator vector scratch space is too small.");

    int[] tensorTargets = new int[decode.tensorsLength()];
    for (int index = 0; index < tensorTargets.length; index++) {
      tensorTargets[index] = tablePosition(decode.tensors(index));
      require(tensorTargets[index]
              > tailTensorVector + Integer.BYTES + index * Integer.BYTES,
          "Decoder tensor table " + index + " is not forward-addressable.");
    }
    int[] operatorTargets = new int[tailOperatorCount];
    for (int index = 0; index < operatorTargets.length; index++) {
      operatorTargets[index] = tablePosition(
          decode.operators(stage.prefixOperatorCount() + index));
      require(operatorTargets[index]
              > tailOperatorVector + Integer.BYTES + index * Integer.BYTES,
          "Decoder operator table " + index + " is not forward-addressable.");
    }

    int signatureVector = tableVectorStart(
        model,
        18,
        label + " signatures");
    require(bytes.getInt(signatureVector) == 2,
        label + " signature vector length changed.");

    return new BoundaryLayout(
        decodeOperatorVector,
        decodeOutputVector + Integer.BYTES,
        decodeSignatureOutputField,
        tailTensorVector,
        tailOperatorVector,
        tailInputField,
        tailOutputField,
        tailInputVector,
        tailOutputVector,
        signatureVector + 2 * Integer.BYTES,
        tensorTargets,
        operatorTargets);
  }

  private static ByteBuffer buildBoundarySignature(BoundaryStage stage) {
    FlatBufferBuilder builder = new FlatBufferBuilder(512);
    int encoderName = builder.createString("encoder_hidden_states");
    int boundaryName = builder.createString(stage.boundaryInputName());
    int maskName = builder.createString("causal_mask");
    int outputName = builder.createString("hidden_pre_layer_norm");
    int signatureKey = builder.createString(stage.tailSignatureKey());
    int encoderMap = TensorMap.createTensorMap(builder, encoderName, 1);
    int boundaryMap = TensorMap.createTensorMap(
        builder,
        boundaryName,
        stage.boundaryTensor());
    int maskMap = TensorMap.createTensorMap(builder, maskName, 0);
    int outputMap = TensorMap.createTensorMap(
        builder,
        outputName,
        PRE_LAYER_NORM_TENSOR_INDEX);
    int inputs = SignatureDef.createInputsVector(
        builder,
        new int[] {encoderMap, boundaryMap, maskMap});
    int outputs = SignatureDef.createOutputsVector(
        builder,
        new int[] {outputMap});
    // SignatureDef slot 3 is the deprecated_tag string in the current TFLite
    // schema. subgraph_index is slot 4; writing the index to slot 3 leaves the
    // native LiteRT loader on its default subgraph (0), even though the older
    // cached Java schema can still parse the other signature fields.
    builder.startTable(5);
    builder.addInt(4, 1, 0);
    builder.addOffset(2, signatureKey, 0);
    builder.addOffset(1, outputs, 0);
    builder.addOffset(0, inputs, 0);
    int signature = builder.endTable();
    builder.finish(signature);
    return builder.dataBuffer().slice().order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void writeForwardOffset(
      FileChannel channel,
      int fieldOrElement,
      int target,
      String label) throws IOException {
    require(target > fieldOrElement,
        label + " must point forward in the FlatBuffer.");
    writeLittleEndianInt(channel, fieldOrElement, target - fieldOrElement);
  }

  private static void generateProjection(
      Path input,
      Path output,
      SourceContract source) throws IOException {
    require(source.weightSize() <= Integer.MAX_VALUE,
        "Projection weight is too large for a FlatBuffer byte vector.");
    ByteBuffer weight = ByteBuffer.allocate((int) source.weightSize());
    try (FileChannel channel = FileChannel.open(input, StandardOpenOption.READ)) {
      readFully(channel, weight, source.weightOffset());
    }
    weight.flip();
    require(source.weightSha256().equals(sha256(weight.duplicate())),
        "Projection weight changed while reading the pinned model.");

    int initialCapacity = Math.toIntExact(
        source.weightSize()
            + (long) source.scales().length * Float.BYTES
            + (long) source.zeroPoints().length * Long.BYTES
            + 2L * 1024L * 1024L);
    FlatBufferBuilder builder = new FlatBufferBuilder(initialCapacity);

    int weightData = Buffer.createDataVector(builder, weight);
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int constantWeightBuffer = Buffer.createBuffer(builder, weightData);
    int buffers = Model.createBuffersVector(
        builder,
        new int[] {emptyBuffer, constantWeightBuffer});

    int scales = QuantizationParameters.createScaleVector(
        builder,
        source.scales());
    int zeroPoints = QuantizationParameters.createZeroPointVector(
        builder,
        source.zeroPoints());
    int weightQuantization = QuantizationParameters.createQuantizationParameters(
        builder,
        0,
        0,
        scales,
        zeroPoints,
        QuantizationDetails.NONE,
        0,
        0);

    int hiddenName = builder.createString("hidden");
    int weightName = builder.createString("vocabulary_projection_weight");
    int logitsName = builder.createString("logits");
    int hiddenShape =
        Tensor.createShapeVector(builder, new int[] {1, HIDDEN_WIDTH});
    int weightShape = Tensor.createShapeVector(
        builder,
        new int[] {VOCABULARY_SIZE, HIDDEN_WIDTH});
    int logitsShape =
        Tensor.createShapeVector(builder, new int[] {1, VOCABULARY_SIZE});

    int hiddenTensor = createRankedTensor(
        builder,
        hiddenShape,
        TensorType.FLOAT32,
        0,
        hiddenName,
        0);
    int weightTensor = createRankedTensor(
        builder,
        weightShape,
        TensorType.INT8,
        1,
        weightName,
        weightQuantization);
    int logitsTensor = createRankedTensor(
        builder,
        logitsShape,
        TensorType.FLOAT32,
        0,
        logitsName,
        0);
    int tensors = SubGraph.createTensorsVector(
        builder,
        new int[] {hiddenTensor, weightTensor, logitsTensor});

    int fullyConnectedOptions = FullyConnectedOptions.createFullyConnectedOptions(
        builder,
        ActivationFunctionType.NONE,
        FullyConnectedOptionsWeightsFormat.DEFAULT,
        true,
        false);
    int operatorInputs =
        Operator.createInputsVector(builder, new int[] {0, 1, -1});
    int operatorOutputs =
        Operator.createOutputsVector(builder, new int[] {2});
    int projectionOperator = Operator.createOperator(
        builder,
        0,
        operatorInputs,
        operatorOutputs,
        BuiltinOptions.FullyConnectedOptions,
        fullyConnectedOptions,
        0,
        CustomOptionsFormat.FLEXBUFFERS,
        0,
        0);
    int operators =
        SubGraph.createOperatorsVector(builder, new int[] {projectionOperator});
    int graphInputs = SubGraph.createInputsVector(builder, new int[] {0});
    int graphOutputs = SubGraph.createOutputsVector(builder, new int[] {2});
    int graphName = builder.createString("project_active_row");
    int subgraph = SubGraph.createSubGraph(
        builder,
        tensors,
        graphInputs,
        graphOutputs,
        operators,
        graphName);
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {subgraph});

    int fullyConnectedCode = OperatorCode.createOperatorCode(
        builder,
        (byte) BuiltinOperator.FULLY_CONNECTED,
        0,
        5,
        BuiltinOperator.FULLY_CONNECTED);
    int operatorCodes =
        Model.createOperatorCodesVector(builder, new int[] {fullyConnectedCode});

    int signatureInputName = builder.createString("hidden");
    int signatureOutputName = builder.createString("logits");
    int signatureInput =
        TensorMap.createTensorMap(builder, signatureInputName, 0);
    int signatureOutput =
        TensorMap.createTensorMap(builder, signatureOutputName, 2);
    int signatureInputs =
        SignatureDef.createInputsVector(builder, new int[] {signatureInput});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {signatureOutput});
    // In this schema artifact, methodName is the field now named signature_key.
    int signatureKey = builder.createString("project");
    int signature = SignatureDef.createSignatureDef(
        builder,
        signatureInputs,
        signatureOutputs,
        signatureKey,
        0);
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});

    int description = builder.createString(
        "VoiceMemo diagnostic active-row Whisper vocabulary projection");
    int model = Model.createModel(
        builder,
        3,
        operatorCodes,
        subgraphs,
        description,
        buffers,
        0,
        0,
        signatures);
    Model.finishModelBuffer(builder, model);

    Path temporary = createSiblingTemporary(output);
    try {
      ByteBuffer finished = builder.dataBuffer().duplicate();
      try (FileChannel channel =
          FileChannel.open(
              temporary,
              StandardOpenOption.WRITE,
              StandardOpenOption.TRUNCATE_EXISTING)) {
        while (finished.hasRemaining()) {
          channel.write(finished);
        }
        channel.force(true);
      }
      Artifact candidate = new Artifact(
          PROJECTION_FILE,
          Files.size(temporary),
          sha256(temporary));
      require(candidate.size() == EXPECTED_PROJECTION_SIZE,
          "Generated projection size changed: " + candidate.size() + ".");
      require(EXPECTED_PROJECTION_SHA256.equals(candidate.sha256()),
          "Generated projection SHA-256 changed: " + candidate.sha256() + ".");
      moveAtomically(temporary, output);
      // As above, keep the pre-move candidate unmapped for Windows and parse
      // the exact, pinned-hash destination after the atomic move.
      verifyProjection(output, source);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void generateLayerNormProjection(
      Path input,
      Path output,
      SourceContract source) throws IOException {
    ByteBuffer weight = readConstantRegion(
        input,
        source.weightOffset(),
        source.weightSize(),
        source.weightSha256(),
        "Projection weight");
    ByteBuffer beta = readConstantRegion(
        input,
        source.beta(),
        "Layer-norm beta");
    ByteBuffer gamma = readConstantRegion(
        input,
        source.gamma(),
        "Layer-norm gamma");
    ByteBuffer epsilon = ByteBuffer.allocate(Float.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(source.epsilonBits());
    epsilon.flip();
    require(EPSILON_SHA256.equals(sha256(epsilon.duplicate())),
        "Layer-norm epsilon changed while preparing the tail model.");
    ByteBuffer reductionAxis = ByteBuffer.allocate(Integer.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(1);
    reductionAxis.flip();

    int initialCapacity = Math.toIntExact(
        source.weightSize()
            + source.beta().size()
            + source.gamma().size()
            + (long) source.scales().length * Float.BYTES
            + (long) source.zeroPoints().length * Long.BYTES
            + 2L * 1024L * 1024L);
    FlatBufferBuilder builder = new FlatBufferBuilder(initialCapacity);

    int weightData = Buffer.createDataVector(builder, weight);
    int gammaData = Buffer.createDataVector(builder, gamma);
    int betaData = Buffer.createDataVector(builder, beta);
    int epsilonData = Buffer.createDataVector(builder, epsilon);
    int reductionAxisData = Buffer.createDataVector(builder, reductionAxis);
    int emptyBuffer = Buffer.createBuffer(builder, 0);
    int betaBuffer = Buffer.createBuffer(builder, betaData);
    int gammaBuffer = Buffer.createBuffer(builder, gammaData);
    int weightBuffer = Buffer.createBuffer(builder, weightData);
    int epsilonBuffer = Buffer.createBuffer(builder, epsilonData);
    int reductionAxisBuffer =
        Buffer.createBuffer(builder, reductionAxisData);
    int buffers = Model.createBuffersVector(
        builder,
        new int[] {
            emptyBuffer,
            betaBuffer,
            gammaBuffer,
            weightBuffer,
            epsilonBuffer,
            reductionAxisBuffer,
        });

    int scales = QuantizationParameters.createScaleVector(
        builder,
        source.scales());
    int zeroPoints = QuantizationParameters.createZeroPointVector(
        builder,
        source.zeroPoints());
    int weightQuantization = QuantizationParameters.createQuantizationParameters(
        builder,
        0,
        0,
        scales,
        zeroPoints,
        QuantizationDetails.NONE,
        0,
        0);

    int[] tensors = new int[TAIL_TENSOR_COUNT];
    tensors[TAIL_INPUT] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "hidden_pre_layer_norm",
        0);
    tensors[TAIL_BETA] = createNamedRankedTensor(
        builder,
        new int[] {HIDDEN_WIDTH},
        TensorType.FLOAT32,
        1,
        "layer_norm_beta",
        0);
    tensors[TAIL_GAMMA] = createNamedRankedTensor(
        builder,
        new int[] {HIDDEN_WIDTH},
        TensorType.FLOAT32,
        2,
        "layer_norm_gamma",
        0);
    tensors[TAIL_WEIGHT] = createNamedRankedTensor(
        builder,
        new int[] {VOCABULARY_SIZE, HIDDEN_WIDTH},
        TensorType.INT8,
        3,
        "vocabulary_projection_weight",
        weightQuantization);
    tensors[TAIL_EPSILON] = createNamedRankedTensor(
        builder,
        new int[] {},
        TensorType.FLOAT32,
        4,
        "layer_norm_epsilon",
        0);
    tensors[TAIL_AXIS] = createNamedRankedTensor(
        builder,
        new int[] {1},
        TensorType.INT32,
        5,
        "layer_norm_axis",
        0);
    tensors[TAIL_MEAN] = createNamedRankedTensor(
        builder,
        new int[] {1, 1},
        TensorType.FLOAT32,
        0,
        "layer_norm_mean",
        0);
    tensors[TAIL_SQUARED_DIFFERENCE] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "layer_norm_squared_difference",
        0);
    tensors[TAIL_VARIANCE] = createNamedRankedTensor(
        builder,
        new int[] {1, 1},
        TensorType.FLOAT32,
        0,
        "layer_norm_variance",
        0);
    tensors[TAIL_VARIANCE_PLUS_EPSILON] = createNamedRankedTensor(
        builder,
        new int[] {1, 1},
        TensorType.FLOAT32,
        0,
        "layer_norm_variance_plus_epsilon",
        0);
    tensors[TAIL_RECIPROCAL_STDDEV] = createNamedRankedTensor(
        builder,
        new int[] {1, 1},
        TensorType.FLOAT32,
        0,
        "layer_norm_reciprocal_stddev",
        0);
    tensors[TAIL_SCALED_GAMMA] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "layer_norm_scaled_gamma",
        0);
    tensors[TAIL_SCALED_INPUT] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "layer_norm_scaled_input",
        0);
    tensors[TAIL_SCALED_MEAN] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "layer_norm_scaled_mean",
        0);
    tensors[TAIL_CENTERING_TERM] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "layer_norm_centering_term",
        0);
    tensors[TAIL_NORMALIZED] = createNamedRankedTensor(
        builder,
        new int[] {1, HIDDEN_WIDTH},
        TensorType.FLOAT32,
        0,
        "hidden_post_layer_norm",
        0);
    tensors[TAIL_LOGITS] = createNamedRankedTensor(
        builder,
        new int[] {1, VOCABULARY_SIZE},
        TensorType.FLOAT32,
        0,
        "logits",
        0);
    int tensorVector = SubGraph.createTensorsVector(builder, tensors);

    int reducerOptions =
        ReducerOptions.createReducerOptions(builder, true);
    int addOptions = AddOptions.createAddOptions(
        builder,
        ActivationFunctionType.NONE,
        true);
    int mulOptions =
        MulOptions.createMulOptions(builder, ActivationFunctionType.NONE);
    int subOptions = SubOptions.createSubOptions(
        builder,
        ActivationFunctionType.NONE,
        true);
    int fullyConnectedOptions = FullyConnectedOptions.createFullyConnectedOptions(
        builder,
        ActivationFunctionType.NONE,
        FullyConnectedOptionsWeightsFormat.DEFAULT,
        true,
        false);

    int[] operators = new int[LAYER_NORM_TAIL_OPERATOR_COUNT];
    operators[0] = createOperator(
        builder,
        TAIL_OPCODE_MEAN,
        new int[] {TAIL_INPUT, TAIL_AXIS},
        new int[] {TAIL_MEAN},
        BuiltinOptions.ReducerOptions,
        reducerOptions);
    operators[1] = createOperator(
        builder,
        TAIL_OPCODE_SQUARED_DIFFERENCE,
        new int[] {TAIL_INPUT, TAIL_MEAN},
        new int[] {TAIL_SQUARED_DIFFERENCE},
        BuiltinOptions.NONE,
        0);
    operators[2] = createOperator(
        builder,
        TAIL_OPCODE_MEAN,
        new int[] {TAIL_SQUARED_DIFFERENCE, TAIL_AXIS},
        new int[] {TAIL_VARIANCE},
        BuiltinOptions.ReducerOptions,
        reducerOptions);
    operators[3] = createOperator(
        builder,
        TAIL_OPCODE_ADD,
        new int[] {TAIL_VARIANCE, TAIL_EPSILON},
        new int[] {TAIL_VARIANCE_PLUS_EPSILON},
        BuiltinOptions.AddOptions,
        addOptions);
    operators[4] = createOperator(
        builder,
        TAIL_OPCODE_RSQRT,
        new int[] {TAIL_VARIANCE_PLUS_EPSILON},
        new int[] {TAIL_RECIPROCAL_STDDEV},
        BuiltinOptions.NONE,
        0);
    operators[5] = createOperator(
        builder,
        TAIL_OPCODE_MUL,
        new int[] {TAIL_RECIPROCAL_STDDEV, TAIL_GAMMA},
        new int[] {TAIL_SCALED_GAMMA},
        BuiltinOptions.MulOptions,
        mulOptions);
    operators[6] = createOperator(
        builder,
        TAIL_OPCODE_MUL,
        new int[] {TAIL_INPUT, TAIL_SCALED_GAMMA},
        new int[] {TAIL_SCALED_INPUT},
        BuiltinOptions.MulOptions,
        mulOptions);
    operators[7] = createOperator(
        builder,
        TAIL_OPCODE_MUL,
        new int[] {TAIL_MEAN, TAIL_SCALED_GAMMA},
        new int[] {TAIL_SCALED_MEAN},
        BuiltinOptions.MulOptions,
        mulOptions);
    operators[8] = createOperator(
        builder,
        TAIL_OPCODE_SUB,
        new int[] {TAIL_BETA, TAIL_SCALED_MEAN},
        new int[] {TAIL_CENTERING_TERM},
        BuiltinOptions.SubOptions,
        subOptions);
    operators[9] = createOperator(
        builder,
        TAIL_OPCODE_ADD,
        new int[] {TAIL_SCALED_INPUT, TAIL_CENTERING_TERM},
        new int[] {TAIL_NORMALIZED},
        BuiltinOptions.AddOptions,
        addOptions);
    operators[10] = createOperator(
        builder,
        TAIL_OPCODE_FULLY_CONNECTED,
        new int[] {TAIL_NORMALIZED, TAIL_WEIGHT, -1},
        new int[] {TAIL_LOGITS},
        BuiltinOptions.FullyConnectedOptions,
        fullyConnectedOptions);
    int operatorVector = SubGraph.createOperatorsVector(builder, operators);

    int graphInputs =
        SubGraph.createInputsVector(builder, new int[] {TAIL_INPUT});
    int graphOutputs =
        SubGraph.createOutputsVector(builder, new int[] {TAIL_LOGITS});
    int graphName =
        builder.createString("normalize_and_project_active_row");
    int subgraph = SubGraph.createSubGraph(
        builder,
        tensorVector,
        graphInputs,
        graphOutputs,
        operatorVector,
        graphName);
    int subgraphs = Model.createSubgraphsVector(builder, new int[] {subgraph});

    int[] operatorCodes = {
        createOperatorCode(builder, BuiltinOperator.ADD, 1),
        createOperatorCode(builder, BuiltinOperator.MEAN, 1),
        createOperatorCode(builder, BuiltinOperator.SQUARED_DIFFERENCE, 1),
        createOperatorCode(builder, BuiltinOperator.RSQRT, 1),
        createOperatorCode(builder, BuiltinOperator.FULLY_CONNECTED, 5),
        createOperatorCode(builder, BuiltinOperator.MUL, 1),
        createOperatorCode(builder, BuiltinOperator.SUB, 1),
    };
    int operatorCodeVector =
        Model.createOperatorCodesVector(builder, operatorCodes);

    int signatureInputName =
        builder.createString("hidden_pre_layer_norm");
    int signatureOutputName = builder.createString("logits");
    int signatureInput = TensorMap.createTensorMap(
        builder,
        signatureInputName,
        TAIL_INPUT);
    int signatureOutput = TensorMap.createTensorMap(
        builder,
        signatureOutputName,
        TAIL_LOGITS);
    int signatureInputs =
        SignatureDef.createInputsVector(builder, new int[] {signatureInput});
    int signatureOutputs =
        SignatureDef.createOutputsVector(builder, new int[] {signatureOutput});
    int signatureKey = builder.createString("normalize_and_project");
    int signature = SignatureDef.createSignatureDef(
        builder,
        signatureInputs,
        signatureOutputs,
        signatureKey,
        0);
    int signatures =
        Model.createSignatureDefsVector(builder, new int[] {signature});

    int description = builder.createString(
        "VoiceMemo diagnostic active-row Whisper layer norm and "
            + "vocabulary projection");
    int model = Model.createModel(
        builder,
        3,
        operatorCodeVector,
        subgraphs,
        description,
        buffers,
        0,
        0,
        signatures);
    Model.finishModelBuffer(builder, model);

    Path temporary = createSiblingTemporary(output);
    try {
      ByteBuffer finished = builder.dataBuffer().duplicate();
      try (FileChannel channel =
          FileChannel.open(
              temporary,
              StandardOpenOption.WRITE,
              StandardOpenOption.TRUNCATE_EXISTING)) {
        while (finished.hasRemaining()) {
          channel.write(finished);
        }
        channel.force(true);
      }
      Artifact candidate = new Artifact(
          LAYER_NORM_PROJECTION_FILE,
          Files.size(temporary),
          sha256(temporary));
      if (EXPECTED_LAYER_NORM_PROJECTION_SIZE >= 0) {
        require(candidate.size() == EXPECTED_LAYER_NORM_PROJECTION_SIZE,
            "Generated layer-norm tail size changed: "
                + candidate.size() + ".");
      }
      if (!EXPECTED_LAYER_NORM_PROJECTION_SHA256.isEmpty()) {
        require(EXPECTED_LAYER_NORM_PROJECTION_SHA256.equals(
                candidate.sha256()),
            "Generated layer-norm tail SHA-256 changed: "
                + candidate.sha256() + ".");
      }
      moveAtomically(temporary, output);
      verifyLayerNormProjection(output, source);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static ByteBuffer readConstantRegion(
      Path input,
      ConstantRegion region,
      String label) throws IOException {
    return readConstantRegion(
        input,
        region.offset(),
        region.size(),
        region.sha256(),
        label);
  }

  private static ByteBuffer readConstantRegion(
      Path input,
      long offset,
      long size,
      String expectedSha256,
      String label) throws IOException {
    require(size <= Integer.MAX_VALUE,
        label + " is too large for a FlatBuffer byte vector.");
    ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(size));
    try (FileChannel channel = FileChannel.open(input, StandardOpenOption.READ)) {
      readFully(channel, bytes, offset);
    }
    bytes.flip();
    require(expectedSha256.equals(sha256(bytes.duplicate())),
        label + " changed while reading the pinned model.");
    return bytes;
  }

  private static int createNamedRankedTensor(
      FlatBufferBuilder builder,
      int[] shape,
      byte type,
      long buffer,
      String name,
      int quantization) {
    int nameOffset = builder.createString(name);
    int shapeOffset = Tensor.createShapeVector(builder, shape);
    return createRankedTensor(
        builder,
        shapeOffset,
        type,
        buffer,
        nameOffset,
        quantization);
  }

  private static int createOperator(
      FlatBufferBuilder builder,
      int opcodeIndex,
      int[] inputs,
      int[] outputs,
      byte builtinOptionsType,
      int builtinOptions) {
    int inputVector = Operator.createInputsVector(builder, inputs);
    int outputVector = Operator.createOutputsVector(builder, outputs);
    return Operator.createOperator(
        builder,
        opcodeIndex,
        inputVector,
        outputVector,
        builtinOptionsType,
        builtinOptions,
        0,
        CustomOptionsFormat.FLEXBUFFERS,
        0,
        0);
  }

  private static int createOperatorCode(
      FlatBufferBuilder builder,
      int builtinCode,
      int version) {
    return OperatorCode.createOperatorCode(
        builder,
        (byte) builtinCode,
        0,
        version,
        builtinCode);
  }

  private static Artifact verifyCore(Path core) throws IOException {
    requireRegularFile(core, "Generated post-layer-norm GPU-core model");
    verifyCoreContract(
        core,
        CORE_DECODE_OPERATOR_COUNT,
        HIDDEN_TENSOR_INDEX,
        CORE_PATCHES,
        "post-layer-norm GPU core");
    Artifact artifact = new Artifact(
        CORE_FILE,
        Files.size(core),
        sha256(core));
    require(artifact.size() == EXPECTED_CORE_SIZE,
        "GPU-core size changed: " + artifact.size() + ".");
    return artifact;
  }

  private static Artifact verifyPreLayerNormCore(Path core) throws IOException {
    requireRegularFile(core, "Generated pre-layer-norm GPU-core model");
    verifyCoreContract(
        core,
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT,
        PRE_LAYER_NORM_TENSOR_INDEX,
        PRE_LAYER_NORM_CORE_PATCHES,
        "pre-layer-norm GPU core");
    Artifact artifact = new Artifact(
        PRE_LAYER_NORM_CORE_FILE,
        Files.size(core),
        sha256(core));
    require(artifact.size() == EXPECTED_PRE_LAYER_NORM_CORE_SIZE,
        "Pre-layer-norm GPU-core size changed: " + artifact.size() + ".");
    return artifact;
  }

  private static Artifact verifyBoundaryModel(
      Path candidate,
      Path source,
      BoundaryStage stage) throws IOException {
    requireRegularFile(candidate, "Generated boundary-split model");
    requireRegularFile(source, "Pinned Whisper model");
    long candidateSize = Files.size(candidate);
    require(candidateSize > ORIGINAL_SIZE
            && candidateSize - ORIGINAL_SIZE < 4_096,
        "Boundary-split model size is outside the compact shared-model range: "
            + candidateSize + ".");
    if (stage.expectedSize() >= 0) {
      require(candidateSize == stage.expectedSize(),
          "Boundary-split model size changed: " + candidateSize + ".");
    }

    try (FileChannel candidateChannel =
            FileChannel.open(candidate, StandardOpenOption.READ);
        FileChannel sourceChannel =
            FileChannel.open(source, StandardOpenOption.READ)) {
      ByteBuffer candidateBytes =
          candidateChannel.map(
                  FileChannel.MapMode.READ_ONLY,
                  0,
                  candidateChannel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      ByteBuffer sourceBytes =
          sourceChannel.map(
                  FileChannel.MapMode.READ_ONLY,
                  0,
                  sourceChannel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model candidateModel = parseModel(
          candidateBytes,
          "Generated " + stage.id() + " boundary model");
      Model sourceModel = parseModel(sourceBytes, "Pinned Whisper source");
      require(candidateModel.subgraphsLength() == 2,
          "Boundary model must retain exactly two shared-buffer subgraphs.");
      require(candidateModel.signatureDefsLength() == 2,
          "Boundary model must retain exactly two signatures.");
      SubGraph prefix = candidateModel.subgraphs(0);
      SubGraph tail = candidateModel.subgraphs(1);
      SubGraph sourceDecode = sourceModel.subgraphs(0);

      require(prefix.operatorsLength() == stage.prefixOperatorCount(),
          "Boundary prefix operator count changed.");
      require(prefix.tensorsLength() == sourceDecode.tensorsLength(),
          "Boundary prefix tensor count changed.");
      require(prefix.inputsLength() == 3
              && prefix.inputs(0) == 0
              && prefix.inputs(1) == 1
              && prefix.inputs(2) == 2,
          "Boundary prefix inputs changed.");
      require(prefix.outputsLength() == 1
              && prefix.outputs(0) == stage.boundaryTensor(),
          "Boundary prefix output changed.");
      require(findSignatureOutput(candidateModel, "decode")
              == stage.boundaryTensor(),
          "Boundary prefix signature output changed.");
      requireTensor(
          prefix.tensors(stage.boundaryTensor()),
          TensorType.FLOAT32,
          new int[] {1, DECODER_LENGTH, HIDDEN_WIDTH},
          "Boundary prefix output");

      int expectedTailOperatorCount =
          PRE_LAYER_NORM_CORE_OPERATOR_COUNT - stage.prefixOperatorCount();
      require(tail.tensorsLength() == sourceDecode.tensorsLength(),
          "Boundary tail tensor count changed.");
      require(tail.operatorsLength() == expectedTailOperatorCount,
          "Boundary tail operator count changed.");
      require(tail.inputsLength() == 3
              && tail.inputs(0) == 1
              && tail.inputs(1) == stage.boundaryTensor()
              && tail.inputs(2) == 0,
          "Boundary tail inputs changed.");
      require(tail.outputsLength() == 1
              && tail.outputs(0) == PRE_LAYER_NORM_TENSOR_INDEX,
          "Boundary tail output changed.");
      requireTensor(
          tail.tensors(PRE_LAYER_NORM_TENSOR_INDEX),
          TensorType.FLOAT32,
          new int[] {1, DECODER_LENGTH, HIDDEN_WIDTH},
          "Boundary tail output");

      Operator firstTail = requireOperator(
          candidateModel,
          tail,
          0,
          BuiltinOperator.MEAN,
          1,
          new int[] {stage.boundaryTensor(), REDUCTION_AXIS_TENSOR_INDEX},
          new int[] {stage.boundaryTensor() + 1},
          "Boundary tail first layer-norm mean");
      requireReducerOptions(firstTail, "Boundary tail first layer-norm mean");
      requireAddOptions(
          requireOperator(
              candidateModel,
              tail,
              expectedTailOperatorCount - 1,
              BuiltinOperator.ADD,
              1,
              new int[] {409, 422},
              new int[] {PRE_LAYER_NORM_TENSOR_INDEX},
              "Boundary tail final decoder residual"),
          "Boundary tail final decoder residual");

      for (int index = 0; index < sourceDecode.tensorsLength(); index++) {
        require(
            tablePosition(tail.tensors(index))
                == tablePosition(sourceDecode.tensors(index)),
            "Boundary tail tensor table " + index
                + " is not shared with the pinned decoder.");
      }
      for (int index = 0; index < stage.prefixOperatorCount(); index++) {
        require(
            tablePosition(prefix.operators(index))
                == tablePosition(sourceDecode.operators(index)),
            "Boundary prefix operator table " + index
                + " changed from the pinned decoder.");
      }
      for (int index = 0; index < expectedTailOperatorCount; index++) {
        require(
            tablePosition(tail.operators(index))
                == tablePosition(sourceDecode.operators(
                    stage.prefixOperatorCount() + index)),
            "Boundary tail operator table " + index
                + " changed from the pinned decoder.");
      }

      SignatureDef tailSignature = findSignature(
          candidateModel,
          stage.tailSignatureKey());
      require(!signatureHasDeprecatedTag(tailSignature),
          "Boundary tail signature deprecated-tag slot must remain absent.");
      require(signatureSubgraphIndex(tailSignature) == 1,
          "Boundary tail signature must select subgraph 1.");
      require(tailSignature.inputsLength() == 3,
          "Boundary tail signature input count changed.");
      requireTensorMap(
          tailSignature.inputs(0),
          "encoder_hidden_states",
          1,
          "Boundary tail encoder input");
      requireTensorMap(
          tailSignature.inputs(1),
          stage.boundaryInputName(),
          stage.boundaryTensor(),
          "Boundary tail residual input");
      requireTensorMap(
          tailSignature.inputs(2),
          "causal_mask",
          0,
          "Boundary tail mask input");
      require(tailSignature.outputsLength() == 1,
          "Boundary tail signature output count changed.");
      requireTensorMap(
          tailSignature.outputs(0),
          "hidden_pre_layer_norm",
          PRE_LAYER_NORM_TENSOR_INDEX,
          "Boundary tail output");

      SourceContract sourceContract = inspectPinnedSource(source);
      require(
          sourceContract.weightSha256().equals(
              sha256Region(
                  candidateChannel,
                  sourceContract.weightOffset(),
                  sourceContract.weightSize())),
          "Boundary model projection bytes changed.");
    }

    return new Artifact(
        stage.fileName(),
        candidateSize,
        sha256(candidate));
  }

  private static void requireTensorMap(
      TensorMap map,
      String expectedName,
      int expectedTensor,
      String label) {
    require(map != null, label + " is missing.");
    require(expectedName.equals(map.name()),
        label + " name changed.");
    require(map.tensorIndex() == expectedTensor,
        label + " tensor index changed.");
  }

  private static int signatureSubgraphIndex(SignatureDef signature) {
    int offset = tableOffset(signature, 12);
    if (offset == 0) return 0;
    return signature.getByteBuffer().getInt(tablePosition(signature) + offset);
  }

  private static boolean signatureHasDeprecatedTag(SignatureDef signature) {
    return tableOffset(signature, 10) != 0;
  }

  private static void verifyCoreContract(
      Path core,
      int expectedOperatorCount,
      int expectedOutputTensor,
      Patch[] patches,
      String label) throws IOException {
    require(Files.size(core) == ORIGINAL_SIZE,
        label + " must remain exactly " + ORIGINAL_SIZE + " bytes.");
    try (FileChannel channel = FileChannel.open(core, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes, "Generated " + label);
      SubGraph decode = model.subgraphs(DECODE_SUBGRAPH);
      require(decode.operatorsLength() == expectedOperatorCount,
          label + " decode operator count changed.");
      require(decode.outputsLength() == 1
              && decode.outputs(0) == expectedOutputTensor,
          label + " graph output is not tensor " + expectedOutputTensor + ".");
      require(findSignatureOutput(model, "decode") == expectedOutputTensor,
          label + " decode signature output is not tensor "
              + expectedOutputTensor + ".");
      requireTensor(
          decode.tensors(expectedOutputTensor),
          TensorType.FLOAT32,
          new int[] {1, DECODER_LENGTH, HIDDEN_WIDTH},
          label + " output");
      if (expectedOutputTensor == PRE_LAYER_NORM_TENSOR_INDEX) {
        requireAddOptions(
            requireOperator(
                model,
                decode,
                PRE_LAYER_NORM_CORE_OPERATOR_COUNT - 1,
                BuiltinOperator.ADD,
                1,
                new int[] {409, 422},
                new int[] {PRE_LAYER_NORM_TENSOR_INDEX},
                "Generated pre-layer-norm residual ADD"),
            "Generated pre-layer-norm residual ADD");
      }
      for (Patch patch : patches) {
        require(readLittleEndianInt(channel, patch.offset())
                == patch.replacement(),
            label + " patch verification failed for "
                + patch.description() + ".");
      }
    }
  }

  private static Artifact verifyProjection(
      Path projection,
      SourceContract source) throws IOException {
    requireRegularFile(projection, "Generated active-row projection model");
    try (FileChannel channel =
        FileChannel.open(projection, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(bytes, "Generated active-row projection model");
      require(model.version() == 3, "Projection model schema version must be 3.");
      require(model.subgraphsLength() == 1,
          "Projection model must contain one subgraph.");
      require(model.operatorCodesLength() == 1,
          "Projection model must contain one operator code.");
      require(model.signatureDefsLength() == 1,
          "Projection model must contain one signature.");
      require(findSignatureOutput(model, "project") == 2,
          "Projection signature output must be tensor 2.");

      SubGraph graph = model.subgraphs(0);
      require("project_active_row".equals(graph.name()),
          "Projection subgraph name changed.");
      require(graph.tensorsLength() == 3
              && graph.operatorsLength() == 1
              && graph.inputsLength() == 1
              && graph.inputs(0) == 0
              && graph.outputsLength() == 1
              && graph.outputs(0) == 2,
          "Projection subgraph topology changed.");
      requireTensor(
          graph.tensors(0),
          TensorType.FLOAT32,
          new int[] {1, HIDDEN_WIDTH},
          "Projection input");
      requireTensor(
          graph.tensors(1),
          TensorType.INT8,
          new int[] {VOCABULARY_SIZE, HIDDEN_WIDTH},
          "Projection weight");
      requireTensor(
          graph.tensors(2),
          TensorType.FLOAT32,
          new int[] {1, VOCABULARY_SIZE},
          "Projection output");
      require(tensorHasRank(graph.tensors(0))
              && tensorHasRank(graph.tensors(1))
              && tensorHasRank(graph.tensors(2)),
          "Projection tensors must be explicitly ranked.");

      Operator operator = graph.operators(0);
      OperatorCode code = model.operatorCodes((int) operator.opcodeIndex());
      require(code.builtinCode() == BuiltinOperator.FULLY_CONNECTED
              && code.version() == 5,
          "Projection operator must be FULLY_CONNECTED version 5.");
      require(operator.inputsLength() == 3
              && operator.inputs(0) == 0
              && operator.inputs(1) == 1
              && operator.inputs(2) == -1
              && operator.outputsLength() == 1
              && operator.outputs(0) == 2,
          "Projection operator bindings changed.");
      FullyConnectedOptions options =
          (FullyConnectedOptions) operator.builtinOptions(
              new FullyConnectedOptions());
      require(options != null
              && options.fusedActivationFunction() == ActivationFunctionType.NONE
              && options.weightsFormat()
                  == FullyConnectedOptionsWeightsFormat.DEFAULT
              && options.keepNumDims()
              && !options.asymmetricQuantizeInputs(),
          "Projection FULLY_CONNECTED options changed.");

      Tensor weight = graph.tensors(1);
      QuantizationParameters quantization = weight.quantization();
      require(quantization != null
              && quantization.scaleLength() == source.scales().length
              && quantization.zeroPointLength() == source.zeroPoints().length
              && quantization.quantizedDimension() == 0
              && quantization.detailsType() == QuantizationDetails.NONE,
          "Projection weight quantization contract changed.");
      for (int index = 0; index < source.scales().length; index++) {
        require(Float.floatToIntBits(quantization.scale(index))
                == Float.floatToIntBits(source.scales()[index]),
            "Projection scale differs at channel " + index + ".");
        require(quantization.zeroPoint(index) == source.zeroPoints()[index],
            "Projection zero point differs at channel " + index + ".");
      }

      Buffer weightBuffer = model.buffers((int) weight.buffer());
      require(weightBuffer.dataLength() == source.weightSize(),
          "Projection embedded weight byte size changed.");
      ByteBuffer embeddedWeight = weightBuffer.dataAsByteBuffer();
      require(embeddedWeight != null,
          "Projection embedded weight buffer is unavailable.");
      ByteBuffer weightSlice = embeddedWeight.slice();
      require(weightSlice.remaining() == source.weightSize(),
          "Projection embedded weight view has the wrong size.");
      require(source.weightSha256().equals(sha256(weightSlice)),
          "Projection embedded weight bytes differ from the pinned model.");
    }

    Artifact artifact = new Artifact(
        PROJECTION_FILE,
        Files.size(projection),
        sha256(projection));
    if (EXPECTED_PROJECTION_SIZE >= 0) {
      require(artifact.size() == EXPECTED_PROJECTION_SIZE,
          "Projection model size changed: " + artifact.size() + ".");
    }
    return artifact;
  }

  private static Artifact verifyLayerNormProjection(
      Path projection,
      SourceContract source) throws IOException {
    requireRegularFile(
        projection,
        "Generated active-row layer-norm projection model");
    try (FileChannel channel =
        FileChannel.open(projection, StandardOpenOption.READ)) {
      ByteBuffer bytes =
          channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
              .order(ByteOrder.LITTLE_ENDIAN);
      Model model = parseModel(
          bytes,
          "Generated active-row layer-norm projection model");
      require(model.subgraphsLength() == 1,
          "Layer-norm tail must contain one subgraph.");
      require(model.operatorCodesLength() == 7,
          "Layer-norm tail must contain seven builtin operator codes.");
      require(model.signatureDefsLength() == 1,
          "Layer-norm tail must contain one signature.");
      SignatureDef signature = model.signatureDefs(0);
      require("normalize_and_project".equals(signature.methodName())
              && signature.inputsLength() == 1
              && "hidden_pre_layer_norm".equals(signature.inputs(0).name())
              && signature.outputsLength() == 1
              && "logits".equals(signature.outputs(0).name()),
          "Layer-norm tail signature names changed.");
      require(findSignatureInput(model, "normalize_and_project") == TAIL_INPUT,
          "Layer-norm tail signature input changed.");
      require(findSignatureOutput(model, "normalize_and_project")
              == TAIL_LOGITS,
          "Layer-norm tail signature output changed.");

      SubGraph graph = model.subgraphs(0);
      require("normalize_and_project_active_row".equals(graph.name()),
          "Layer-norm tail subgraph name changed.");
      require(graph.tensorsLength() == TAIL_TENSOR_COUNT
              && graph.operatorsLength() == LAYER_NORM_TAIL_OPERATOR_COUNT
              && graph.inputsLength() == 1
              && graph.inputs(0) == TAIL_INPUT
              && graph.outputsLength() == 1
              && graph.outputs(0) == TAIL_LOGITS,
          "Layer-norm tail topology changed.");

      byte[] types = {
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.INT8,
          TensorType.FLOAT32,
          TensorType.INT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
          TensorType.FLOAT32,
      };
      int[][] shapes = {
          {1, HIDDEN_WIDTH},
          {HIDDEN_WIDTH},
          {HIDDEN_WIDTH},
          {VOCABULARY_SIZE, HIDDEN_WIDTH},
          {},
          {1},
          {1, 1},
          {1, HIDDEN_WIDTH},
          {1, 1},
          {1, 1},
          {1, 1},
          {1, HIDDEN_WIDTH},
          {1, HIDDEN_WIDTH},
          {1, HIDDEN_WIDTH},
          {1, HIDDEN_WIDTH},
          {1, HIDDEN_WIDTH},
          {1, VOCABULARY_SIZE},
      };
      long[] buffers = {
          0, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      };
      for (int index = 0; index < TAIL_TENSOR_COUNT; index++) {
        Tensor tensor = graph.tensors(index);
        requireTensor(
            tensor,
            types[index],
            shapes[index],
            "Layer-norm tail tensor " + index);
        require(tensorHasRank(tensor),
            "Layer-norm tail tensor " + index + " is not explicitly ranked.");
        require(tensor.buffer() == buffers[index],
            "Layer-norm tail tensor " + index + " buffer changed.");
      }

      requireEmbeddedBuffer(
          model,
          graph.tensors(TAIL_BETA),
          source.beta().size(),
          source.beta().sha256(),
          "Layer-norm tail beta");
      requireEmbeddedBuffer(
          model,
          graph.tensors(TAIL_GAMMA),
          source.gamma().size(),
          source.gamma().sha256(),
          "Layer-norm tail gamma");
      requireEmbeddedBuffer(
          model,
          graph.tensors(TAIL_WEIGHT),
          source.weightSize(),
          source.weightSha256(),
          "Layer-norm tail projection weight");
      ByteBuffer epsilon = requireEmbeddedBuffer(
          model,
          graph.tensors(TAIL_EPSILON),
          Float.BYTES,
          EPSILON_SHA256,
          "Layer-norm tail epsilon");
      require(epsilon.order(ByteOrder.LITTLE_ENDIAN).getInt(0)
              == source.epsilonBits(),
          "Layer-norm tail epsilon value changed.");
      ByteBuffer axis = requireEmbeddedBuffer(
          model,
          graph.tensors(TAIL_AXIS),
          Integer.BYTES,
          null,
          "Layer-norm tail axis");
      require(axis.order(ByteOrder.LITTLE_ENDIAN).getInt(0) == 1,
          "Active-row layer norm must reduce decoder dimension 1.");
      requireWeightQuantization(
          graph.tensors(TAIL_WEIGHT),
          source,
          "Layer-norm tail projection weight");

      requireOperatorCode(
          model, TAIL_OPCODE_ADD, BuiltinOperator.ADD, 1, "ADD");
      requireOperatorCode(
          model, TAIL_OPCODE_MEAN, BuiltinOperator.MEAN, 1, "MEAN");
      requireOperatorCode(
          model,
          TAIL_OPCODE_SQUARED_DIFFERENCE,
          BuiltinOperator.SQUARED_DIFFERENCE,
          1,
          "SQUARED_DIFFERENCE");
      requireOperatorCode(
          model, TAIL_OPCODE_RSQRT, BuiltinOperator.RSQRT, 1, "RSQRT");
      requireOperatorCode(
          model,
          TAIL_OPCODE_FULLY_CONNECTED,
          BuiltinOperator.FULLY_CONNECTED,
          5,
          "FULLY_CONNECTED");
      requireOperatorCode(
          model, TAIL_OPCODE_MUL, BuiltinOperator.MUL, 1, "MUL");
      requireOperatorCode(
          model, TAIL_OPCODE_SUB, BuiltinOperator.SUB, 1, "SUB");

      requireReducerOptions(
          requireOperator(
              model,
              graph,
              0,
              BuiltinOperator.MEAN,
              1,
              new int[] {TAIL_INPUT, TAIL_AXIS},
              new int[] {TAIL_MEAN},
              "Active-row layer-norm mean"),
          "Active-row layer-norm mean");
      requireNoOptions(
          requireOperator(
              model,
              graph,
              1,
              BuiltinOperator.SQUARED_DIFFERENCE,
              1,
              new int[] {TAIL_INPUT, TAIL_MEAN},
              new int[] {TAIL_SQUARED_DIFFERENCE},
              "Active-row layer-norm squared difference"),
          "Active-row layer-norm squared difference");
      requireReducerOptions(
          requireOperator(
              model,
              graph,
              2,
              BuiltinOperator.MEAN,
              1,
              new int[] {TAIL_SQUARED_DIFFERENCE, TAIL_AXIS},
              new int[] {TAIL_VARIANCE},
              "Active-row layer-norm variance"),
          "Active-row layer-norm variance");
      requireAddOptions(
          requireOperator(
              model,
              graph,
              3,
              BuiltinOperator.ADD,
              1,
              new int[] {TAIL_VARIANCE, TAIL_EPSILON},
              new int[] {TAIL_VARIANCE_PLUS_EPSILON},
              "Active-row layer-norm epsilon ADD"),
          "Active-row layer-norm epsilon ADD");
      requireNoOptions(
          requireOperator(
              model,
              graph,
              4,
              BuiltinOperator.RSQRT,
              1,
              new int[] {TAIL_VARIANCE_PLUS_EPSILON},
              new int[] {TAIL_RECIPROCAL_STDDEV},
              "Active-row layer-norm reciprocal square root"),
          "Active-row layer-norm reciprocal square root");
      requireMulOptions(
          requireOperator(
              model,
              graph,
              5,
              BuiltinOperator.MUL,
              1,
              new int[] {TAIL_RECIPROCAL_STDDEV, TAIL_GAMMA},
              new int[] {TAIL_SCALED_GAMMA},
              "Active-row layer-norm scaled gamma"),
          "Active-row layer-norm scaled gamma");
      requireMulOptions(
          requireOperator(
              model,
              graph,
              6,
              BuiltinOperator.MUL,
              1,
              new int[] {TAIL_INPUT, TAIL_SCALED_GAMMA},
              new int[] {TAIL_SCALED_INPUT},
              "Active-row layer-norm scaled input"),
          "Active-row layer-norm scaled input");
      requireMulOptions(
          requireOperator(
              model,
              graph,
              7,
              BuiltinOperator.MUL,
              1,
              new int[] {TAIL_MEAN, TAIL_SCALED_GAMMA},
              new int[] {TAIL_SCALED_MEAN},
              "Active-row layer-norm scaled mean"),
          "Active-row layer-norm scaled mean");
      requireSubOptions(
          requireOperator(
              model,
              graph,
              8,
              BuiltinOperator.SUB,
              1,
              new int[] {TAIL_BETA, TAIL_SCALED_MEAN},
              new int[] {TAIL_CENTERING_TERM},
              "Active-row layer-norm centering term"),
          "Active-row layer-norm centering term");
      requireAddOptions(
          requireOperator(
              model,
              graph,
              9,
              BuiltinOperator.ADD,
              1,
              new int[] {TAIL_SCALED_INPUT, TAIL_CENTERING_TERM},
              new int[] {TAIL_NORMALIZED},
              "Active-row layer-norm output ADD"),
          "Active-row layer-norm output ADD");
      Operator fullyConnected = requireOperator(
          model,
          graph,
          10,
          BuiltinOperator.FULLY_CONNECTED,
          5,
          new int[] {TAIL_NORMALIZED, TAIL_WEIGHT, -1},
          new int[] {TAIL_LOGITS},
          "Active-row vocabulary projection");
      require(fullyConnected.builtinOptionsType()
              == BuiltinOptions.FullyConnectedOptions,
          "Active-row vocabulary projection options type changed.");
      FullyConnectedOptions fullyConnectedOptions =
          (FullyConnectedOptions) fullyConnected.builtinOptions(
              new FullyConnectedOptions());
      require(fullyConnectedOptions != null
              && fullyConnectedOptions.fusedActivationFunction()
                  == ActivationFunctionType.NONE
              && fullyConnectedOptions.weightsFormat()
                  == FullyConnectedOptionsWeightsFormat.DEFAULT
              && fullyConnectedOptions.keepNumDims()
              && !fullyConnectedOptions.asymmetricQuantizeInputs(),
          "Active-row vocabulary projection options changed.");
    }

    Artifact artifact = new Artifact(
        LAYER_NORM_PROJECTION_FILE,
        Files.size(projection),
        sha256(projection));
    if (EXPECTED_LAYER_NORM_PROJECTION_SIZE >= 0) {
      require(artifact.size() == EXPECTED_LAYER_NORM_PROJECTION_SIZE,
          "Layer-norm tail size changed: " + artifact.size() + ".");
    }
    return artifact;
  }

  private static ByteBuffer requireEmbeddedBuffer(
      Model model,
      Tensor tensor,
      long expectedSize,
      String expectedSha256,
      String label) {
    Buffer buffer = model.buffers((int) tensor.buffer());
    require(buffer != null && buffer.dataLength() == expectedSize,
        label + " embedded byte size changed.");
    ByteBuffer bytes = buffer.dataAsByteBuffer();
    require(bytes != null && bytes.remaining() == expectedSize,
        label + " embedded bytes are unavailable.");
    ByteBuffer slice = bytes.slice();
    if (expectedSha256 != null) {
      require(expectedSha256.equals(sha256(slice.duplicate())),
          label + " bytes differ from the pinned model.");
    }
    return slice.order(ByteOrder.LITTLE_ENDIAN);
  }

  private static void requireWeightQuantization(
      Tensor weight,
      SourceContract source,
      String label) {
    QuantizationParameters quantization = weight.quantization();
    require(quantization != null
            && quantization.scaleLength() == source.scales().length
            && quantization.zeroPointLength() == source.zeroPoints().length
            && quantization.quantizedDimension() == 0
            && quantization.detailsType() == QuantizationDetails.NONE,
        label + " quantization contract changed.");
    for (int index = 0; index < source.scales().length; index++) {
      require(Float.floatToIntBits(quantization.scale(index))
              == Float.floatToIntBits(source.scales()[index]),
          label + " scale differs at channel " + index + ".");
      require(quantization.zeroPoint(index) == source.zeroPoints()[index],
          label + " zero point differs at channel " + index + ".");
    }
  }

  private static void requireOperatorCode(
      Model model,
      int index,
      int builtinCode,
      int version,
      String label) {
    OperatorCode code = model.operatorCodes(index);
    require(code != null
            && code.builtinCode() == builtinCode
            && code.version() == version,
        "Layer-norm tail " + label + " opcode changed.");
  }

  private static boolean isValidCore(Path core) {
    try {
      return Files.isRegularFile(core)
          && Files.size(core) == EXPECTED_CORE_SIZE
          && EXPECTED_CORE_SHA256.equals(sha256(core));
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static boolean isValidProjection(Path projection) {
    try {
      return Files.isRegularFile(projection)
          && Files.size(projection) == EXPECTED_PROJECTION_SIZE
          && EXPECTED_PROJECTION_SHA256.equals(sha256(projection));
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static boolean isValidPreLayerNormCore(Path core) {
    try {
      return Files.isRegularFile(core)
          && Files.size(core) == EXPECTED_PRE_LAYER_NORM_CORE_SIZE
          && !EXPECTED_PRE_LAYER_NORM_CORE_SHA256.isEmpty()
          && EXPECTED_PRE_LAYER_NORM_CORE_SHA256.equals(sha256(core));
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static boolean isValidLayerNormProjection(Path projection) {
    try {
      return Files.isRegularFile(projection)
          && EXPECTED_LAYER_NORM_PROJECTION_SIZE >= 0
          && Files.size(projection) == EXPECTED_LAYER_NORM_PROJECTION_SIZE
          && !EXPECTED_LAYER_NORM_PROJECTION_SHA256.isEmpty()
          && EXPECTED_LAYER_NORM_PROJECTION_SHA256.equals(sha256(projection));
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static boolean isValidBoundaryModel(
      Path model,
      Path source,
      BoundaryStage stage) {
    try {
      if (!Files.isRegularFile(model)) return false;
      if (stage.expectedSize() >= 0
          && Files.size(model) != stage.expectedSize()) {
        return false;
      }
      if (!stage.expectedSha256().isEmpty()
          && !stage.expectedSha256().equals(sha256(model))) {
        return false;
      }
      verifyBoundaryModel(model, source, stage);
      return true;
    } catch (Throwable ignored) {
      return false;
    }
  }

  private static void verifyPinnedGeneratedHashes(
      Artifact core,
      Artifact projection,
      Artifact preLayerNormCore,
      Artifact layerNormProjection) {
    if (!EXPECTED_CORE_SHA256.isEmpty()) {
      require(EXPECTED_CORE_SHA256.equals(core.sha256()),
          "GPU-core SHA-256 changed: " + core.sha256() + ".");
    }
    if (!EXPECTED_PROJECTION_SHA256.isEmpty()) {
      require(EXPECTED_PROJECTION_SHA256.equals(projection.sha256()),
          "Projection SHA-256 changed: " + projection.sha256() + ".");
    }
    if (!EXPECTED_PRE_LAYER_NORM_CORE_SHA256.isEmpty()) {
      require(EXPECTED_PRE_LAYER_NORM_CORE_SHA256.equals(
              preLayerNormCore.sha256()),
          "Pre-layer-norm GPU-core SHA-256 changed: "
              + preLayerNormCore.sha256() + ".");
    }
    if (!EXPECTED_LAYER_NORM_PROJECTION_SHA256.isEmpty()) {
      require(EXPECTED_LAYER_NORM_PROJECTION_SHA256.equals(
              layerNormProjection.sha256()),
          "Layer-norm tail SHA-256 changed: "
              + layerNormProjection.sha256() + ".");
    }
  }

  private static void verifyPinnedBoundaryHash(
      Artifact artifact,
      BoundaryStage stage) {
    if (stage.expectedSize() >= 0) {
      require(artifact.size() == stage.expectedSize(),
          "Boundary model size changed: " + artifact.size() + ".");
    }
    if (!stage.expectedSha256().isEmpty()) {
      require(stage.expectedSha256().equals(artifact.sha256()),
          "Boundary model SHA-256 changed: " + artifact.sha256() + ".");
    }
  }

  private static void writeManifest(
      Path manifest,
      SourceContract source,
      Artifact core,
      Artifact projection,
      Artifact preLayerNormCore,
      Artifact layerNormProjection) throws IOException {
    String json = """
        {
          "schema_version": 2,
          "diagnostic_only": true,
          "source": {
            "size_bytes": %d,
            "sha256": "%s"
          },
          "gpu_core": {
            "file": "%s",
            "size_bytes": %d,
            "sha256": "%s",
            "decode_output_tensor": %d,
            "decode_output_shape": [1, %d, %d],
            "decode_operator_count": %d
          },
          "active_row_projection": {
            "file": "%s",
            "size_bytes": %d,
            "sha256": "%s",
            "input_shape": [1, %d],
            "output_shape": [1, %d],
            "weight_shape": [%d, %d],
            "weight_size_bytes": %d,
            "weight_sha256": "%s",
            "quantized_dimension": 0,
            "scale_count": %d,
            "zero_point_count": %d,
            "scale_min": %s,
            "scale_max": %s
          },
          "pre_layer_norm_gpu_core": {
            "file": "%s",
            "size_bytes": %d,
            "sha256": "%s",
            "decode_output_tensor": %d,
            "decode_output_shape": [1, %d, %d],
            "decode_operator_count": %d
          },
          "active_row_layer_norm_projection": {
            "file": "%s",
            "size_bytes": %d,
            "sha256": "%s",
            "signature": "normalize_and_project",
            "input_name": "hidden_pre_layer_norm",
            "input_shape": [1, %d],
            "output_name": "logits",
            "output_shape": [1, %d],
            "operator_count": %d,
            "builtin_only_cpu_control": true,
            "xnnpack_with_builtin_fallback": true,
            "epsilon": %s,
            "beta_size_bytes": %d,
            "beta_sha256": "%s",
            "gamma_size_bytes": %d,
            "gamma_sha256": "%s",
            "weight_shape": [%d, %d],
            "weight_size_bytes": %d,
            "weight_sha256": "%s"
          }
        }
        """.formatted(
        source.size(),
        source.sha256(),
        core.fileName(),
        core.size(),
        core.sha256(),
        HIDDEN_TENSOR_INDEX,
        DECODER_LENGTH,
        HIDDEN_WIDTH,
        CORE_DECODE_OPERATOR_COUNT,
        projection.fileName(),
        projection.size(),
        projection.sha256(),
        HIDDEN_WIDTH,
        VOCABULARY_SIZE,
        VOCABULARY_SIZE,
        HIDDEN_WIDTH,
        source.weightSize(),
        source.weightSha256(),
        source.scales().length,
        source.zeroPoints().length,
        Float.toString(source.minimumScale()),
        Float.toString(source.maximumScale()),
        preLayerNormCore.fileName(),
        preLayerNormCore.size(),
        preLayerNormCore.sha256(),
        PRE_LAYER_NORM_TENSOR_INDEX,
        DECODER_LENGTH,
        HIDDEN_WIDTH,
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT,
        layerNormProjection.fileName(),
        layerNormProjection.size(),
        layerNormProjection.sha256(),
        HIDDEN_WIDTH,
        VOCABULARY_SIZE,
        LAYER_NORM_TAIL_OPERATOR_COUNT,
        Float.toString(Float.intBitsToFloat(source.epsilonBits())),
        source.beta().size(),
        source.beta().sha256(),
        source.gamma().size(),
        source.gamma().sha256(),
        VOCABULARY_SIZE,
        HIDDEN_WIDTH,
        source.weightSize(),
        source.weightSha256());
    Path temporary = createSiblingTemporary(manifest);
    try {
      Files.writeString(
          temporary,
          json,
          StandardCharsets.UTF_8,
          StandardOpenOption.WRITE,
          StandardOpenOption.TRUNCATE_EXISTING);
      moveAtomically(temporary, manifest);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void printSummary(
      Path input,
      Path outputDirectory,
      SourceContract source,
      Artifact core,
      Artifact projection,
      Artifact preLayerNormCore,
      Artifact layerNormProjection) {
    System.out.println("source=" + input.toAbsolutePath().normalize());
    System.out.println("source_sha256=" + source.sha256());
    System.out.println("output_directory="
        + outputDirectory.toAbsolutePath().normalize());
    System.out.println("gpu_core_size=" + core.size());
    System.out.println("gpu_core_sha256=" + core.sha256());
    System.out.println("projection_size=" + projection.size());
    System.out.println("projection_sha256=" + projection.sha256());
    System.out.println(
        "pre_layer_norm_gpu_core_size=" + preLayerNormCore.size());
    System.out.println(
        "pre_layer_norm_gpu_core_sha256=" + preLayerNormCore.sha256());
    System.out.println(
        "layer_norm_projection_size=" + layerNormProjection.size());
    System.out.println(
        "layer_norm_projection_sha256=" + layerNormProjection.sha256());
    System.out.println("layer_norm_beta_sha256=" + source.beta().sha256());
    System.out.println("layer_norm_gamma_sha256=" + source.gamma().sha256());
    System.out.println("projection_weight_sha256=" + source.weightSha256());
    System.out.println("verified=true");
  }

  private static void writeBoundaryManifest(
      Path manifest,
      SourceContract source,
      BoundaryStage stage,
      Artifact artifact) throws IOException {
    String json = """
        {
          "schema_version": 1,
          "diagnostic_only": true,
          "stage": "%s",
          "source": {
            "size_bytes": %d,
            "sha256": "%s"
          },
          "shared_model": {
            "file": "%s",
            "size_bytes": %d,
            "sha256": "%s",
            "production_encode_subgraph_repurposed": true,
            "decoder_constants_shared_between_subgraphs": true
          },
          "prefix": {
            "backend": "%s",
            "signature": "decode",
            "operator_range_inclusive": [0, %d],
            "operator_count": %d,
            "output_tensor": %d,
            "output_name": "%s",
            "output_shape": [1, %d, %d]
          },
          "remaining_blocks_tail": {
            "backend": "%s",
            "signature": "%s",
            "operator_range_inclusive": [%d, %d],
            "operator_count": %d,
            "input_names": [
              "encoder_hidden_states",
              "%s",
              "causal_mask"
            ],
            "input_tensor_indices": [1, %d, 0],
            "output_name": "hidden_pre_layer_norm",
            "output_tensor": %d,
            "output_shape": [1, %d, %d],
            "causal_mask_is_preserved_for_signature_compatibility_but_unused_by_pinned_graph": true
          },
          "active_row_tail": {
            "reuse_file": "%s",
            "sha256": "%s",
            "signature": "normalize_and_project",
            "input_shape": [1, %d],
            "output_shape": [1, %d],
            "select_active_row_before_invocation": true
          }
        }
        """.formatted(
        stage.id(),
        source.size(),
        source.sha256(),
        artifact.fileName(),
        artifact.size(),
        artifact.sha256(),
        stage.prefixBackend(),
        stage.prefixOperatorCount() - 1,
        stage.prefixOperatorCount(),
        stage.boundaryTensor(),
        stage.boundaryInputName(),
        DECODER_LENGTH,
        HIDDEN_WIDTH,
        stage.tailBackend(),
        stage.tailSignatureKey(),
        stage.prefixOperatorCount(),
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT - 1,
        PRE_LAYER_NORM_CORE_OPERATOR_COUNT - stage.prefixOperatorCount(),
        stage.boundaryInputName(),
        stage.boundaryTensor(),
        PRE_LAYER_NORM_TENSOR_INDEX,
        DECODER_LENGTH,
        HIDDEN_WIDTH,
        LAYER_NORM_PROJECTION_FILE,
        EXPECTED_LAYER_NORM_PROJECTION_SHA256,
        HIDDEN_WIDTH,
        VOCABULARY_SIZE);
    Path temporary = createSiblingTemporary(manifest);
    try {
      Files.writeString(
          temporary,
          json,
          StandardCharsets.UTF_8,
          StandardOpenOption.WRITE,
          StandardOpenOption.TRUNCATE_EXISTING);
      moveAtomically(temporary, manifest);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static void printBoundarySummary(
      Path input,
      Path outputDirectory,
      BoundaryStage stage,
      Artifact artifact) {
    System.out.println("source=" + input.toAbsolutePath().normalize());
    System.out.println("output_directory="
        + outputDirectory.toAbsolutePath().normalize());
    System.out.println("stage=" + stage.id());
    System.out.println("shared_model_file=" + artifact.fileName());
    System.out.println("shared_model_size=" + artifact.size());
    System.out.println("shared_model_sha256=" + artifact.sha256());
    System.out.println("prefix_backend=" + stage.prefixBackend());
    System.out.println(
        "prefix_operator_range=0-" + (stage.prefixOperatorCount() - 1));
    System.out.println("prefix_output_tensor=" + stage.boundaryTensor());
    System.out.println("tail_backend=" + stage.tailBackend());
    System.out.println("tail_operator_range="
        + stage.prefixOperatorCount() + "-"
        + (PRE_LAYER_NORM_CORE_OPERATOR_COUNT - 1));
    System.out.println("tail_output_tensor=" + PRE_LAYER_NORM_TENSOR_INDEX);
    System.out.println(
        "active_row_tail_sha256=" + EXPECTED_LAYER_NORM_PROJECTION_SHA256);
    System.out.println("verified=true");
  }

  private static void selfTest() throws IOException {
    require("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            .equals(sha256(ByteBuffer.wrap("abc".getBytes(StandardCharsets.UTF_8)))),
        "SHA-256 self-test failed.");
    Path directory = Files.createTempDirectory("whisper-decoder-split-self-test-");
    Path file = directory.resolve("patch.bin");
    try {
      ByteBuffer bytes = ByteBuffer.allocate(16).order(ByteOrder.LITTLE_ENDIAN);
      bytes.putInt(4, 321);
      Files.write(file, bytes.array());
      try (FileChannel channel =
          FileChannel.open(
              file,
              StandardOpenOption.READ,
              StandardOpenOption.WRITE)) {
        require(readLittleEndianInt(channel, 4) == 321,
            "Patch read self-test failed.");
        writeLittleEndianInt(channel, 4, 320);
        require(readLittleEndianInt(channel, 4) == 320,
            "Patch write self-test failed.");
      }
      BoundaryStage stage = findBoundaryStage("after-3-blocks");
      require(stage.prefixOperatorCount() == 233
              && stage.boundaryTensor() == 346,
          "Boundary-stage lookup self-test failed.");
      BoundaryStage afterTwoBlocks = findBoundaryStage("after-2-blocks");
      require(afterTwoBlocks.prefixOperatorCount() == 156
              && afterTwoBlocks.boundaryTensor() == 269
              && afterTwoBlocks.expectedSize() == 755_273_920L
              && "f5926743061104e49dd12f532fe3d35a3025f791bc1ec75c4e2a0b7eef136337"
                  .equals(afterTwoBlocks.expectedSha256()),
          "Pinned after-two-block boundary identity self-test failed.");
      BoundaryStage afterOneBlock = findBoundaryStage("after-1-block");
      require(afterOneBlock.prefixOperatorCount() == 79
              && afterOneBlock.boundaryTensor() == 192
              && afterOneBlock.expectedSize() == 755_273_920L
              && "ed01a9e67f3a5991580bc7892ec9f747d63ed587bab02a6ef1b5c72366d4a6f3"
                  .equals(afterOneBlock.expectedSha256()),
          "Pinned after-one-block boundary identity self-test failed.");
      BoundaryStage afterCrossAttention =
          findBoundaryStage("after-cross-attention");
      require(afterCrossAttention.prefixOperatorCount() == 65
              && afterCrossAttention.boundaryTensor() == 178
              && "hidden_after_cross_attention"
                  .equals(afterCrossAttention.boundaryInputName())
              && afterCrossAttention.expectedSize() == 755_273_928L
              && "bf71f075f3dffee2d7d233fffc0d77559e624941424a24933830e94386df3f5a"
                  .equals(afterCrossAttention.expectedSha256()),
          "Pinned after-cross-attention boundary identity self-test failed.");
      BoundaryStage afterPositionAdd =
          findBoundaryStage("after-position-add");
      require(afterPositionAdd.prefixOperatorCount() == 2
              && afterPositionAdd.boundaryTensor() == 115
              && "hidden_after_position_add"
                  .equals(afterPositionAdd.boundaryInputName())
              && "gpu".equals(afterPositionAdd.prefixBackend())
              && "cpu".equals(afterPositionAdd.tailBackend())
              && afterPositionAdd.expectedSize() == 755_273_920L
              && "ee557aa9924a66f0b54f70c80759804b137e77dbaa9661d3b12bb5291615495d"
                  .equals(afterPositionAdd.expectedSha256()),
          "Pinned after-position-add boundary identity self-test failed.");
      ByteBuffer signatureBytes = buildBoundarySignature(stage);
      SignatureDef signature = SignatureDef.getRootAsSignatureDef(
          signatureBytes.duplicate().order(ByteOrder.LITTLE_ENDIAN));
      require(stage.tailSignatureKey().equals(signature.methodName()),
          "Boundary signature-key self-test failed.");
      require(!signatureHasDeprecatedTag(signature),
          "Boundary signature deprecated-tag slot must remain absent.");
      require(signatureSubgraphIndex(signature) == 1,
          "Boundary signature subgraph self-test failed.");
      require(signature.inputsLength() == 3
              && signature.inputs(0).tensorIndex() == 1
              && signature.inputs(1).tensorIndex() == stage.boundaryTensor()
              && signature.inputs(2).tensorIndex() == 0,
          "Boundary signature input self-test failed.");
      require(signature.outputsLength() == 1
              && signature.outputs(0).tensorIndex()
                  == PRE_LAYER_NORM_TENSOR_INDEX,
          "Boundary signature output self-test failed.");
      System.out.println("self_test_passed=true");
    } finally {
      Files.deleteIfExists(file);
      Files.deleteIfExists(directory);
    }
  }

  private static BoundaryStage findBoundaryStage(String id) {
    for (BoundaryStage stage : BOUNDARY_STAGES) {
      if (stage.id().equals(id)) return stage;
    }
    throw new ToolFailure("Unsupported boundary stage: " + id + ".");
  }

  private static Model parseModel(ByteBuffer bytes, String label) {
    require(Model.ModelBufferHasIdentifier(bytes.duplicate()),
        label + " does not have the TFL3 identifier.");
    Model model = Model.getRootAsModel(bytes);
    require(model.version() == 3, label + " schema version is not 3.");
    return model;
  }

  private static int findSignatureOutput(Model model, String key) {
    SignatureDef signature = findSignature(model, key);
    require(signature.outputsLength() == 1,
        "Signature " + key + " must have one output.");
    TensorMap output = signature.outputs(0);
    require(output != null, "Signature " + key + " output is missing.");
    return Math.toIntExact(output.tensorIndex());
  }

  private static int findSignatureInput(Model model, String key) {
    SignatureDef signature = findSignature(model, key);
    require(signature.inputsLength() == 1,
        "Signature " + key + " must have one input.");
    TensorMap input = signature.inputs(0);
    require(input != null, "Signature " + key + " input is missing.");
    return Math.toIntExact(input.tensorIndex());
  }

  private static SignatureDef findSignature(Model model, String key) {
    for (int index = 0; index < model.signatureDefsLength(); index++) {
      SignatureDef signature = model.signatureDefs(index);
      // The cached schema artifact calls current signature_key methodName.
      if (key.equals(signature.methodName())) return signature;
    }
    throw new ToolFailure("Required signature is missing: " + key + ".");
  }

  private static void requireTensor(
      Tensor tensor,
      byte expectedType,
      int[] expectedShape,
      String label) {
    require(tensor != null, label + " is missing.");
    require(tensor.type() == expectedType,
        label + " type is " + (tensor.type() & 0xff)
            + "; expected " + (expectedType & 0xff) + ".");
    int[] actualShape = new int[tensor.shapeLength()];
    for (int index = 0; index < actualShape.length; index++) {
      actualShape[index] = tensor.shape(index);
    }
    require(Arrays.equals(actualShape, expectedShape),
        label + " shape is " + Arrays.toString(actualShape)
            + "; expected " + Arrays.toString(expectedShape) + ".");
  }

  private static int createRankedTensor(
      FlatBufferBuilder builder,
      int shape,
      byte type,
      long buffer,
      int name,
      int quantization) {
    // tensorflow-lite-metadata 0.2.0 provides the stable base schema classes,
    // but predates Tensor.has_rank. Emit the append-only current field
    // directly so LiteRT sees these non-scalar tensors as explicitly ranked.
    builder.startTable(CURRENT_TENSOR_FIELD_COUNT);
    builder.addBoolean(TENSOR_HAS_RANK_FIELD, true, false);
    builder.addOffset(4, quantization, 0);
    builder.addOffset(3, name, 0);
    builder.addInt(2, Math.toIntExact(buffer), 0);
    builder.addByte(1, type, 0);
    builder.addOffset(0, shape, 0);
    return builder.endTable();
  }

  private static boolean tensorHasRank(Tensor tensor) {
    int offset = tableOffset(tensor, TENSOR_HAS_RANK_VTABLE_OFFSET);
    return offset != 0
        && tensor.getByteBuffer().get(tablePosition(tensor) + offset) != 0;
  }

  private static Path requireProjectBuildOutput(
      Path projectRoot,
      Path output) throws IOException {
    Path realProjectRoot = projectRoot.toRealPath();
    Path lexicalBuild = realProjectRoot.resolve("build");
    if (!Files.exists(lexicalBuild)) {
      Files.createDirectory(lexicalBuild);
    }
    Path realBuild = lexicalBuild.toRealPath();
    require(realBuild.equals(lexicalBuild),
        "The project build/ directory must not be a symbolic link.");
    Path prospectiveOutput = prospectiveRealPath(output);
    require(prospectiveOutput.startsWith(realBuild),
        "Diagnostic outputs must stay under this project's ignored build/ "
            + "directory: " + realBuild + ".");
    return realBuild;
  }

  private static Path prospectiveRealPath(Path path) throws IOException {
    Path absolute = path.toAbsolutePath().normalize();
    Path existing = absolute;
    while (existing != null && !Files.exists(existing)) {
      existing = existing.getParent();
    }
    require(existing != null,
        "Could not resolve diagnostic output path: " + path + ".");
    Path resolved = existing.toRealPath();
    return resolved.resolve(existing.relativize(absolute)).normalize();
  }

  private static void rejectProductionOverwrite(Path input, Path output)
      throws IOException {
    Path realInput = input.toRealPath();
    Path realOutput = output.toRealPath();
    require(!realOutput.startsWith(realInput.getParent()),
        "Output directory must not be the production model asset directory "
            + "or one of its descendants.");
    require(!realOutput.resolve(CORE_FILE).equals(realInput)
            && !realOutput.resolve(PROJECTION_FILE).equals(realInput)
            && !realOutput.resolve(PRE_LAYER_NORM_CORE_FILE).equals(realInput)
            && !realOutput.resolve(LAYER_NORM_PROJECTION_FILE).equals(realInput),
        "Generated output must not replace the pinned production model.");
  }

  private static void requireRegularFile(Path path, String label) {
    require(Files.isRegularFile(path), label + " is missing: " + path + ".");
  }

  private static void require(boolean condition, String message) {
    if (!condition) throw new ToolFailure(message);
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
      throw new ToolFailure("Could not inspect FlatBuffer table field.", error);
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

  private static int readLittleEndianInt(FileChannel channel, long offset)
      throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(Integer.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, bytes, offset);
    bytes.flip();
    return bytes.getInt();
  }

  private static void writeLittleEndianInt(
      FileChannel channel,
      long offset,
      int value) throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(Integer.BYTES)
        .order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(value).flip();
    writeFully(channel, bytes, offset);
  }

  private static void readFully(
      FileChannel channel,
      ByteBuffer destination,
      long offset) throws IOException {
    long position = offset;
    while (destination.hasRemaining()) {
      int read = channel.read(destination, position);
      if (read < 0) {
        throw new ToolFailure("Unexpected end of file at byte " + position + ".");
      }
      if (read == 0) continue;
      position += read;
    }
  }

  private static void writeFully(
      FileChannel channel,
      ByteBuffer source,
      long offset) throws IOException {
    long position = offset;
    while (source.hasRemaining()) {
      int written = channel.write(source, position);
      if (written == 0) continue;
      position += written;
    }
  }

  private static String sha256(Path path) throws IOException {
    MessageDigest digest = sha256Digest();
    ByteBuffer bytes = ByteBuffer.allocateDirect(8 * 1024 * 1024);
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      while (channel.read(bytes) >= 0) {
        bytes.flip();
        digest.update(bytes);
        bytes.clear();
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static String sha256Region(
      FileChannel channel,
      long offset,
      long length) throws IOException {
    MessageDigest digest = sha256Digest();
    ByteBuffer bytes = ByteBuffer.allocateDirect(8 * 1024 * 1024);
    long position = offset;
    long remaining = length;
    while (remaining > 0) {
      bytes.clear();
      bytes.limit((int) Math.min(bytes.capacity(), remaining));
      int read = channel.read(bytes, position);
      if (read < 0) {
        throw new ToolFailure("Unexpected end of projection weight buffer.");
      }
      if (read == 0) continue;
      bytes.flip();
      digest.update(bytes);
      position += read;
      remaining -= read;
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static String sha256(ByteBuffer source) {
    MessageDigest digest = sha256Digest();
    digest.update(source);
    return HexFormat.of().formatHex(digest.digest());
  }

  private static MessageDigest sha256Digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException error) {
      throw new AssertionError(error);
    }
  }

  private static Path createSiblingTemporary(Path output) throws IOException {
    return Files.createTempFile(
        output.getParent(),
        output.getFileName() + ".",
        ".part");
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

  private record Patch(
      long offset,
      int expected,
      int replacement,
      String description) {}

  private record SourceContract(
      long size,
      String sha256,
      long weightOffset,
      long weightSize,
      String weightSha256,
      float[] scales,
      long[] zeroPoints,
      float minimumScale,
      float maximumScale,
      ConstantRegion beta,
      ConstantRegion gamma,
      int epsilonBits) {}

  private record ConstantRegion(long offset, long size, String sha256) {}

  private record Artifact(String fileName, long size, String sha256) {}

  private record BoundaryStage(
      String id,
      String fileName,
      int prefixOperatorCount,
      int boundaryTensor,
      String boundaryInputName,
      String prefixBackend,
      String tailBackend,
      long expectedSize,
      String expectedSha256) {
    String tailSignatureKey() {
      return "decode_tail_" + id.replace('-', '_');
    }
  }

  private record BoundaryLayout(
      int decodeOperatorVector,
      int decodeOutputElement,
      int decodeSignatureOutputField,
      int tailTensorVector,
      int tailOperatorVector,
      int tailInputField,
      int tailOutputField,
      int tailInputVector,
      int tailOutputVector,
      int secondSignatureVectorElement,
      int[] decodeTensorTargets,
      int[] decodeTailOperatorTargets) {}

  private record Arguments(
      String command,
      Path projectRoot,
      Path input,
      Path output,
      boolean force,
      String stage) {
    static Arguments parse(String[] arguments) {
      if (arguments.length == 0) {
        throw new ToolFailure(
            "Command is required: generate, verify, or self-test.");
      }
      String command = arguments[0];
      if ("self-test".equals(command)) {
        require(arguments.length == 1,
            "self-test does not accept additional arguments.");
        return new Arguments(command, null, null, null, false, null);
      }
      Map<String, String> values = new LinkedHashMap<>();
      boolean force = false;
      for (int index = 1; index < arguments.length; index++) {
        String argument = arguments[index];
        if ("--force".equals(argument)) {
          force = true;
          continue;
        }
        require(argument.startsWith("--"),
            "Unexpected argument: " + argument + ".");
        require(index + 1 < arguments.length,
            "Missing value for " + argument + ".");
        values.put(argument, arguments[++index]);
      }
      boolean boundaryCommand = "generate-boundary".equals(command)
          || "verify-boundary".equals(command);
      require("generate".equals(command)
              || "verify".equals(command)
              || boundaryCommand,
          "Unsupported command: " + command + ".");
      require(values.containsKey("--project-root"),
          "--project-root is required.");
      require(values.containsKey("--input"), "--input is required.");
      require(values.containsKey("--output"), "--output is required.");
      if (boundaryCommand) {
        require(values.containsKey("--stage"),
            "--stage is required for boundary commands.");
        require(values.size() == 4,
            "Boundary commands support only --project-root, --input, "
                + "--output, --stage, and --force.");
      } else {
        require(!values.containsKey("--stage"),
            "--stage is only valid with boundary commands.");
        require(values.size() == 3,
            "Only --project-root, --input, --output, and --force are supported.");
      }
      require(!("verify".equals(command)
              || "verify-boundary".equals(command))
              || !force,
          "--force is only valid with generate commands.");
      return new Arguments(
          command,
          Path.of(values.get("--project-root")),
          Path.of(values.get("--input")),
          Path.of(values.get("--output")),
          force,
          values.get("--stage"));
    }

    Path requiredProjectRoot() {
      if (projectRoot == null) {
        throw new ToolFailure("--project-root is required.");
      }
      return projectRoot.toAbsolutePath().normalize();
    }

    Path requiredInput() {
      if (input == null) throw new ToolFailure("--input is required.");
      return input.toAbsolutePath().normalize();
    }

    Path requiredOutput() {
      if (output == null) throw new ToolFailure("--output is required.");
      return output.toAbsolutePath().normalize();
    }

    BoundaryStage requiredBoundaryStage() {
      if (stage == null) throw new ToolFailure("--stage is required.");
      try {
        return findBoundaryStage(stage);
      } catch (ToolFailure ignored) {
        throw new ToolFailure(
            "Unsupported boundary stage: " + stage
                + ". Expected after-position-add, after-cross-attention, "
                + "after-1-block, after-2-blocks, or after-3-blocks.");
      }
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
