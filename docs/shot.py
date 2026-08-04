import sys, pathlib
from playwright.sync_api import sync_playwright
url = pathlib.Path(sys.argv[1]).resolve().as_uri()
pages = [int(x) for x in sys.argv[2].split(',')]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 794, "height": 1123}, device_scale_factor=1.4)
    pg.goto(url, wait_until="networkidle")
    for n in pages:
        pg.locator(".page").nth(n-1).screenshot(path=f"/tmp/claude-0/-home-user-sp2l/b78b227e-baa8-5db0-b80e-600a15b07c1c/scratchpad/pg{n}.png")
    b.close()
print("ok")
