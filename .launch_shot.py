import pathlib
from playwright.sync_api import sync_playwright

url = "file://" + str(pathlib.Path("index.html").resolve())
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                                args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    page.goto(url)
    page.wait_for_timeout(2500)
    page.screenshot(path=".index_launch_light.png", full_page=True)
    # toggle dark mode
    page.click("#theme-toggle")
    page.wait_for_timeout(1200)
    page.screenshot(path=".index_launch_dark.png", full_page=True)
    print("title:", page.title())
    print("theme:", page.get_attribute("html", "data-theme"))
    browser.close()
print("done")
