# The first release is a sideload build and does not minify. Keep LiteRT's
# signature runner discoverable if shrinking is enabled later.
-keep class org.tensorflow.lite.** { *; }
