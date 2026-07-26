# VoiceMemo agent instructions

## Preserve user recordings

Never uninstall or delete the VoiceMemo app from a connected device as part of development, testing, or troubleshooting. The app stores recordings in app-private storage, so uninstalling it can permanently delete the user's recordings. Use non-destructive diagnostics and in-place updates instead. If an uninstall is ever necessary, stop and obtain explicit user approval after confirming the recordings have been safely exported or backed up.
