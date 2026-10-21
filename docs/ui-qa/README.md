# Control panel visual QA (#14)

Before/after captures from the headless visual QA pass on the restyled control panel. Each image puts the `main` build (before) next to this branch (after).

The panel ran against a loopback fixture backend: the real `register_settings_page` routes and static files, owned settings and HOME/XDG roots, a fixture song library of small generated MP3s, and fake playback (`dj_invoke`) and voice status. Nothing reached a robot or a provider. Headless Chromium drove it at 1440x900, 820x1180 and 390x844, with reduced-motion and light-scheme variants. A CDP screencast recorded every compositor frame (about 50 fps), and an in-page sampler logged element geometry on each animation frame.

| Image | What it shows |
|---|---|
| `01-tablet-deck.jpg` | At 820px the deck readout was about 90px wide, so the state copy wrapped one word per line. The pads now sit under the screen below 1220px. |
| `02-masthead-reconnect.jpg` | When "Reconnect voice" appeared, the side column wrapped under the brand and pushed the page down 75px. The button now opens to the left of the pill. |
| `03-error-copy.jpg` | A proxy 502 page showed `Unexpected token '<'... is not valid JSON` and the library shimmered forever. Now the message says HTTP 502 and the library loads anyway. |
| `04-phone-toasts.jpg` | Five identical messages stacked three copies. A repeat now pulses the newest toast in place. |
| `05-phone-generating-title.jpg` | "Composing, then playing" wrapped to two lines on phones and grew the sticky deck. |
| `06-phone-320-pads.jpg` | At 320px the transport pads overflowed the deck. |
| `07-delete-frames.jpg` | Deleting a row left a 142px hole, then everything below snapped up in one frame. Rows now collapse smoothly. |
| `08-stop-equalizer-frames.jpg` | Stop dropped every equalizer bar to zero and replayed the boot wake-up. The bars now settle in place. |
| `09-playing-row.jpg` | The playing row's button gained a spinner for the whole track, which widened it and rewrapped the filename. |
| `10-library-error.jpg` | A failed first library load kept the skeleton shimmering. It now shows a retry hint. |
