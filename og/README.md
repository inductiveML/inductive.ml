# Social preview images

Each `*.html` here is the source for a 1200×630 PNG in `public/og/`, the size social platforms expect.

Render one with headless Chrome from the project root:

```sh
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=1 --virtual-time-budget=10000 --window-size=1200,630 \
  --screenshot=public/og/<name>.png "file://$PWD/og/<name>.html"
```

Pages pass the image to `Layout` through its `og` prop.
