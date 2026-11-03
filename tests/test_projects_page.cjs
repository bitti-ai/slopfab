// Optional real-browser check. Supply a local Playwright package directory,
// generated project index.html, and an installed Chromium/Chrome executable.
// node tests/test_projects_page.cjs PLAYWRIGHT_DIR INDEX_HTML CHROME_EXE
const { chromium } = require(require("path").resolve(process.argv[2]));
const { pathToFileURL } = require("url");
const path = require("path");
const assert = require("assert/strict");
(async () => {
  const browser = await chromium.launch({ executablePath: process.argv[4], headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(pathToFileURL(path.resolve(process.argv[3])).href);
    await page.getByRole("button", { name: "Review Final film", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("video")?.readyState >= 1);
    const player = page.locator("video");
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("video").currentTime > 0.15);
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    assert(await player.evaluate((m) => m.paused));
    await page.getByRole("slider", { name: "Playback position" }).fill("500");
    assert(Math.abs((await player.evaluate((m) => m.currentTime)) - 2) < 0.25);
    await page.getByRole("combobox", { name: "Playback speed" }).selectOption("1.5");
    assert.equal(await player.evaluate((m) => m.playbackRate), 1.5);
    await page.getByRole("button", { name: "Mute", exact: true }).click();
    assert(await player.evaluate((m) => m.muted));
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "desktop-review.png"),
      fullPage: true,
    });
    await page.getByRole("button", { name: "Review Opening shot", exact: true }).click();
    await page.getByLabel("Action", { exact: true }).selectOption("remove");
    await page
      .getByLabel("Your direction", { exact: false })
      .fill("Keep the music continuous across the cut.");
    await page.getByRole("button", { name: "Create agent prompt" }).click();
    const prompt = await page.getByLabel("Prompt to copy", { exact: true }).inputValue();
    assert(prompt.includes("clip-a") && prompt.includes("Remove this item"));
    assert(
      prompt.includes("rebuild the parent video") && prompt.includes("Keep the music continuous"),
    );
    assert(prompt.includes("project.json") && prompt.includes("exact manifest paths"));
    // Force the file:// clipboard fallback, which must leave a selectable prompt.
    await page.evaluate(() =>
      Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true }),
    );
    await page.getByRole("button", { name: "Copy prompt", exact: true }).click();
    assert((await page.locator("#copy-status").textContent()).length > 0);
    await page.getByRole("button", { name: "Close", exact: true }).click();
    await page.getByRole("button", { name: "Review Cover image", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("#stage img")?.naturalWidth > 0);
    await page.getByRole("button", { name: "Review Night score", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("audio")?.readyState >= 1);
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForFunction(() => document.querySelector("audio").currentTime > 0.15);
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    await page.getByRole("tab", { name: "Clips", exact: true }).click();
    assert.equal(await page.locator("#grid .card").count(), 2);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "mobile-review.png"),
      fullPage: true,
    });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert(await page.locator('.player-controls').evaluate(n =>
      n.getBoundingClientRect().right <= n.parentElement.getBoundingClientRect().right + 1));
    // Malicious metadata is text, never executable markup.
    await page.evaluate(() => {
      project.items[0].title = '<img src=x onerror="window.injected=1">';
      choose(project.items[0]);
    });
    assert.equal(await page.locator("#item-title img").count(), 0);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    assert.deepEqual(errors, []);
    console.log(
      "PASS: offline video/audio/image previews, controls, clip prompts, clipboard fallback, escaping and mobile layout",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
