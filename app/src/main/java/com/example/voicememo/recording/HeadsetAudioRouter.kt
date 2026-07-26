package com.example.voicememo.recording

import android.media.AudioDeviceInfo
import android.media.AudioManager
import android.os.Build
import android.util.Log

internal data class AudioInputRoute(
    val preferredInput: AudioDeviceInfo?,
    val communicationDeviceSelected: Boolean,
    val previousAudioMode: Int,
)

internal class HeadsetAudioRouter(
    private val audioManager: AudioManager,
) {
    fun prepare(): AudioInputRoute {
        val preferredInput = audioManager
            .getDevices(AudioManager.GET_DEVICES_INPUTS)
            .filter(AudioDeviceInfo::isSource)
            .maxByOrNull { HeadsetRoutePolicy.externalInputPriority(it.type) }
            ?.takeIf { HeadsetRoutePolicy.externalInputPriority(it.type) > 0 }

        val previousMode = audioManager.mode
        var communicationDeviceSelected = false
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val communicationDevice = audioManager.availableCommunicationDevices
                .maxByOrNull { HeadsetRoutePolicy.communicationOutputPriority(it.type) }
                ?.takeIf { HeadsetRoutePolicy.communicationOutputPriority(it.type) > 0 }
            if (communicationDevice != null) {
                audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
                communicationDeviceSelected = audioManager.setCommunicationDevice(communicationDevice)
                if (!communicationDeviceSelected) {
                    audioManager.mode = previousMode
                }
            }
        }

        Log.i(
            TAG,
            "Prepared recording route: preferredInputType=${preferredInput?.type}, " +
                "communicationDeviceSelected=$communicationDeviceSelected",
        )
        return AudioInputRoute(
            preferredInput = preferredInput,
            communicationDeviceSelected = communicationDeviceSelected,
            previousAudioMode = previousMode,
        )
    }

    fun clear(route: AudioInputRoute?) {
        route ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S &&
            route.communicationDeviceSelected
        ) {
            audioManager.clearCommunicationDevice()
            audioManager.mode = route.previousAudioMode
        }
    }

    private companion object {
        const val TAG = "HeadsetAudioRouter"
    }
}

internal object HeadsetRoutePolicy {
    fun externalInputPriority(type: Int): Int = when (type) {
        AudioDeviceInfo.TYPE_BLE_HEADSET -> 500
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> 490
        AudioDeviceInfo.TYPE_WIRED_HEADSET -> 400
        AudioDeviceInfo.TYPE_USB_HEADSET -> 390
        AudioDeviceInfo.TYPE_USB_DEVICE -> 380
        else -> 0
    }

    fun communicationOutputPriority(type: Int): Int = when (type) {
        AudioDeviceInfo.TYPE_BLE_HEADSET -> 500
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> 490
        AudioDeviceInfo.TYPE_WIRED_HEADSET -> 400
        AudioDeviceInfo.TYPE_USB_HEADSET -> 390
        else -> 0
    }
}
