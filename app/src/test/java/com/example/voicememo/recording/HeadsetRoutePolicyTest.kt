package com.example.voicememo.recording

import android.media.AudioDeviceInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class HeadsetRoutePolicyTest {
    @Test
    fun `external headset inputs are preferred over the built in microphone`() {
        val builtIn = HeadsetRoutePolicy.externalInputPriority(
            AudioDeviceInfo.TYPE_BUILTIN_MIC,
        )

        assertTrue(
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_BLUETOOTH_SCO) >
                builtIn,
        )
        assertTrue(
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_BLE_HEADSET) >
                builtIn,
        )
        assertTrue(
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_WIRED_HEADSET) >
                builtIn,
        )
        assertTrue(
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_USB_HEADSET) >
                builtIn,
        )
    }

    @Test
    fun `headphones without a microphone are not selected as an input`() {
        assertEquals(
            0,
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_WIRED_HEADPHONES),
        )
        assertEquals(
            0,
            HeadsetRoutePolicy.externalInputPriority(AudioDeviceInfo.TYPE_BLUETOOTH_A2DP),
        )
    }
}
