# Phones, tablets and other computers

The Web UI runs on the computer that generates, and you can use it from any browser on your network: a phone on the sofa, an iPad, or a laptop while the desktop does the work.

## Open it from another device

Start the Web UI so other devices can reach it:

```bash
ziv ui --host local
```

It prints this computer's network address, for example `http://192.168.1.20:8080/`. Open that address on the other device.

!!! warning "Your network only"
    `--host local` makes the Web UI reachable by every device on your network, with no login. Use it on a network you trust.

The server only answers requests addressed to an IP address, `localhost`, the `--host` value, or this computer's name (bare and with `.local`), and refuses changes requested by other websites. To reach it under another name, list that name in `ZIV_UI_ALLOWED_HOSTS`:

```bash
ZIV_UI_ALLOWED_HOSTS=studio.lan ziv ui --host local
```

## Keep it running

```bash
ziv ui --background
```

starts the Web UI, prints its address and gives you the prompt back. It keeps running after you close the terminal; its output goes to the [error log](../concepts/data-folder.md#error-log).

```bash
ziv ui --background-stop
```

stops every running Web UI, including ones running in another terminal, and any job they are running.

## On a phone

![The Workspace on a phone: the preview fills the screen, setting chips and the prompt beneath](../assets/screenshots/placeholder.svg "phone-workspace")

On the phone, Share → *Add to Home Screen* installs the Web UI as an app that opens without the browser's bars.

The preview is the screen:

- **Swipe sideways** through earlier outputs, and **swipe up** for the Gallery.
- **Tap** the preview to open the viewer. Pinch or double-tap to zoom.
- **Press and hold** any image, in the preview, the Gallery strip, the Gallery or the viewer, for its menu.

Under the preview, a strip shows one chip per setting with its value. Tap a chip to change that setting, or **All** for every setting. The first chip is the [preset](workspace.md#presets). With MiniMax H3, Image to Video has a **Last** chip beside **First**.

The prompt sits at the bottom beside **Generate**. Tap it to edit, with the same tabs, enhancer and prompt-file controls as on a computer. Press and hold **Generate**, or tap the arrow beside it, to run with a new seed.

While a job runs, a bar over the preview shows its progress with **Pause**. **⋯** on the bar opens the job's details, with **Skip**, **Repeat** and **Cancel**.

The menu at the top left opens the other pages. Menus, dialogs and sheets close when you pull them down, tap outside them or tap ✕.

### The viewer on a phone

Tapping an image grows it out of its tile into the viewer, which fills the screen.

- **Swipe sideways** and the next image follows your finger.
- **Pull down** and the image shrinks back into its tile.
- **Swipe up** for its details, which rise as a sheet under the image.
- **Tap the image** to hide the bar and the strip, and tap again to bring them back.
- **Pinch or double-tap** to zoom, and **press and hold** for the image's menu.

In the Gallery, a tile shows only its **⋯** button; a tap opens the image. Tap the page's title in the top bar to scroll back to the top.

## On a tablet

Held upright, an iPad shows the preview and the Gallery strip across the top, with *Compose* and *Settings* side by side under them. Held sideways, it has the computer layout.

Any touch screen gets finger-sized controls, and numbers get − and + buttons (hold to repeat) in place of the label dragging a mouse uses.
