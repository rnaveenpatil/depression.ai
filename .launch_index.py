import pathlib
from playwright.sync_api import sync_playwright

root = pathlib.Path("/home/patil/depression.ai")
url = (root / "index.html").as_uri()

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(1500)
    light = root / ".index_launch_light.png"
    page.screenshot(path=str(light), full_page=True)
    # toggle dark
    page.click("#theme-toggle")
    page.wait_for_timeout(800)
    dark = root / ".index_launch_dark.png"
    page.screenshot(path=str(dark), full_page=True)
    print("title:", page.title())
    print("theme:", page.get_attribute("html", "data-theme"))
    browser.close()

for f in (light, dark):
    print(f, f.stat().st_size, "bytes")
