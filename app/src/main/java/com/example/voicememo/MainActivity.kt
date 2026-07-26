package com.example.voicememo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import com.example.voicememo.ui.VoiceMemoApp
import com.example.voicememo.ui.VoiceMemoDestination
import com.example.voicememo.ui.VoiceMemoUiAction
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val viewModel: VoiceMemoViewModel by viewModels()
    private var startRecordingAfterPermission = false

    private val audioPickerLauncher = registerForActivityResult(
        ActivityResultContracts.OpenMultipleDocuments(),
    ) { uris ->
        viewModel.importAudio(uris)
    }

    private val backupExportLauncher = registerForActivityResult(
        ActivityResultContracts.CreateDocument(BACKUP_MIME_TYPE),
    ) { uri ->
        viewModel.exportBackupTo(uri)
    }

    private val backupRestoreLauncher = registerForActivityResult(
        ActivityResultContracts.OpenDocument(),
    ) { uri ->
        viewModel.restoreBackupFrom(uri)
    }

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { results ->
        val microphoneGranted = results[Manifest.permission.RECORD_AUDIO] == true ||
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO,
            ) == PackageManager.PERMISSION_GRANTED
        if (startRecordingAfterPermission && microphoneGranted) {
            viewModel.onAction(VoiceMemoUiAction.StartRecording)
        } else if (startRecordingAfterPermission) {
            Toast.makeText(
                this,
                "Microphone permission is required to record.",
                Toast.LENGTH_LONG,
            ).show()
        }
        startRecordingAfterPermission = false
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        lifecycleScope.launch {
            repeatOnLifecycle(androidx.lifecycle.Lifecycle.State.STARTED) {
                viewModel.effects.collect { effect ->
                    when (effect) {
                        VoiceMemoUiEffect.LaunchAudioPicker -> runCatching {
                            audioPickerLauncher.launch(arrayOf("audio/*"))
                        }.onFailure {
                            viewModel.importAudio(emptyList())
                            Toast.makeText(
                                this@MainActivity,
                                "No compatible document picker is available.",
                                Toast.LENGTH_LONG,
                            ).show()
                        }

                        is VoiceMemoUiEffect.LaunchBackupExport -> runCatching {
                            backupExportLauncher.launch(effect.suggestedFileName)
                        }.onFailure {
                            viewModel.exportBackupTo(null)
                            Toast.makeText(
                                this@MainActivity,
                                "No compatible document provider is available.",
                                Toast.LENGTH_LONG,
                            ).show()
                        }

                        VoiceMemoUiEffect.LaunchBackupRestore -> runCatching {
                            backupRestoreLauncher.launch(BACKUP_MIME_TYPES)
                        }.onFailure {
                            viewModel.restoreBackupFrom(null)
                            Toast.makeText(
                                this@MainActivity,
                                "No compatible document provider is available.",
                                Toast.LENGTH_LONG,
                            ).show()
                        }

                        is VoiceMemoUiEffect.LaunchIntent -> runCatching {
                            startActivity(effect.intent)
                        }.onFailure {
                            Toast.makeText(
                                this@MainActivity,
                                "No compatible app is available.",
                                Toast.LENGTH_LONG,
                            ).show()
                        }
                        is VoiceMemoUiEffect.Message -> Toast.makeText(
                            this@MainActivity,
                            effect.text,
                            Toast.LENGTH_LONG,
                        ).show()
                    }
                }
            }
        }

        setContent {
            val uiState by viewModel.uiState.collectAsStateWithLifecycle()
            val canNavigateBack = uiState.selectedMemoIds.isNotEmpty() ||
                uiState.destination != VoiceMemoDestination.Library
            BackHandler(enabled = canNavigateBack) {
                viewModel.onAction(VoiceMemoUiAction.NavigateBack)
            }
            VoiceMemoApp(
                uiState = uiState,
                onAction = ::dispatch,
            )
        }
    }

    private fun dispatch(action: VoiceMemoUiAction) {
        if (action != VoiceMemoUiAction.StartRecording) {
            viewModel.onAction(action)
            return
        }
        val microphoneGranted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.RECORD_AUDIO,
        ) == PackageManager.PERMISSION_GRANTED
        val notificationGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.POST_NOTIFICATIONS,
            ) == PackageManager.PERMISSION_GRANTED
        if (microphoneGranted && notificationGranted) {
            viewModel.onAction(action)
            return
        }

        startRecordingAfterPermission = true
        permissionLauncher.launch(
            buildList {
                if (!microphoneGranted) add(Manifest.permission.RECORD_AUDIO)
                if (!notificationGranted) {
                    add(Manifest.permission.POST_NOTIFICATIONS)
                }
            }.toTypedArray(),
        )
    }

    private companion object {
        const val BACKUP_MIME_TYPE = "application/zip"
        val BACKUP_MIME_TYPES = arrayOf(
            BACKUP_MIME_TYPE,
            "application/x-zip-compressed",
            "application/octet-stream",
        )
    }
}
