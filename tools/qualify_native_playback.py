"""Qualify SDK mixer continuity with synthetic devices and a delayed PCM burst.

Run this in a Reachy SDK 1.12 GStreamer-capable interpreter. It opens no robot,
camera, microphone, speaker or provider connection. --baseline must fail on the
unconfigured SDK. The configured application must keep the signal continuous.
This verifies software playback scheduling, not acoustic speaker quality.
"""

import sys
import argparse
import subprocess
from pathlib import Path


_PROBE = r'''
import sys,time,json,importlib.util
import numpy as np
import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst
from reachy_mini.media.audio_gstreamer import GStreamerAudio
Gst.init(None)

class SyntheticAudio(GStreamerAudio):
    def _init_pipeline_record(self,pipeline):
        source=Gst.ElementFactory.make('audiotestsrc')
        source.set_property('is-live',True)
        source.set_property('wave',4)
        source.set_property('samplesperbuffer',256)
        caps=Gst.ElementFactory.make('capsfilter')
        caps.set_property('caps',Gst.Caps.from_string(f'audio/x-raw,rate={self.SAMPLE_RATE},channels={self.CHANNELS},format=F32LE,layout=interleaved'))
        self._appsink_audio=Gst.ElementFactory.make('appsink')
        self._appsink_audio.set_property('drop',True)
        self._appsink_audio.set_property('max-buffers',4)
        for element in [source,caps,self._appsink_audio]:pipeline.add(element)
        assert source.link(caps) and caps.link(self._appsink_audio)
    def _build_audiosink_element(self):
        sink=Gst.ElementFactory.make('fakesink','measured_output')
        sink.set_property('sync',True)
        return sink

signal=[]
def observe(pad,info):
    buffer=info.get_buffer();caps=pad.get_current_caps()
    if buffer is None or caps is None:return Gst.PadProbeReturn.OK
    structure=caps.get_structure(0)
    fmt=structure.get_value('format');channels=structure.get_value('channels');rate=structure.get_value('rate')
    dtype={'F32LE':'<f4','S16LE':'<i2','S32LE':'<i4'}.get(fmt)
    if dtype:
        samples=np.frombuffer(buffer.extract_dup(0,buffer.get_size()),dtype=dtype).astype(np.float64)
        if fmt=='S16LE':samples/=32768
        elif fmt=='S32LE':samples/=2147483648
        if samples.size and np.sqrt(np.mean(samples*samples))>.005:
            signal.append((time.monotonic(),samples.size/channels/rate))
    return Gst.PadProbeReturn.OK

audio=SyntheticAudio(log_level='ERROR')
try:
    if sys.argv[1]!='baseline':
        if sys.argv[2]:
            spec=importlib.util.spec_from_file_location('candidate_native_playback',sys.argv[2])
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            configure=module.configure_native_playback
        else:
            from dj_reachy.audio.native_playback import configure_native_playback as configure
        assert configure(audio)>=.5
    audio._pipeline.get_by_name('measured_output').get_static_pad('sink').add_probe(Gst.PadProbeType.BUFFER,observe)
    audio.start_playing()
    time.sleep(.5)
    samples=np.arange(320,dtype=np.float32)
    wave=(np.sin(2*np.pi*1000*samples/audio.SAMPLE_RATE)*.08).astype(np.float32)
    frame=np.column_stack([wave,wave])
    epoch=time.monotonic()
    for index in range(250):
        time.sleep(max(0,epoch+index*.02-time.monotonic()))
        if index==50:time.sleep(.25)
        audio.push_audio_sample(frame)
    time.sleep(1.)
finally:
    audio.stop_playing()
    audio._loop.quit()
    audio._thread_bus_calls.join(3)
max_gap=max((b[0]-a[0] for a,b in zip(signal,signal[1:])),default=0)
seconds=sum(row[1] for row in signal)
print(json.dumps({'mode':sys.argv[1],'max_signal_gap_s':max_gap,'signal_seconds':seconds,'signal_buffers':len(signal),'physical_devices':False,'provider_requests':0,'native_thread_stopped':not audio._thread_bus_calls.is_alive()}))
assert not audio._thread_bus_calls.is_alive()
assert 4.95<=seconds<=5.06,'The rendered signal lost or duplicated PCM'
assert max_gap<.05,'The rendered signal stuttered after the delayed delivery'
'''


def main() -> int:
    """Exercise the real SDK in an isolated process, preserving its exit result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Verify the unconfigured SDK fails the continuity gate")
    parser.add_argument("--candidate-module", type=Path, help="Use an explicit candidate module before package installation")
    args = parser.parse_args()
    mode = "baseline" if args.baseline else "configured"
    module = str(args.candidate_module) if args.candidate_module is not None else ""
    return subprocess.run([sys.executable, "-c", _PROBE, mode, module], check=False, timeout=20).returncode


if __name__ == "__main__":
    raise SystemExit(main())
