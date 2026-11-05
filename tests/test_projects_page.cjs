// Optional real-browser check. Supply a local Playwright package directory,
// generated project index.html, and an installed Chromium/Chrome executable.
// node tests/test_projects_page.cjs PLAYWRIGHT_DIR INDEX_HTML CHROME_EXE
const { chromium } = require(require("path").resolve(process.argv[2]));
const { pathToFileURL } = require("url");
const path = require("path");
const assert = require("assert/strict");
(async () => {
  const browser = await chromium.launch({
    executablePath: process.argv[4],
    headless: true,
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1100 },
    });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(
      pathToFileURL(path.resolve(process.argv[3])).href + "#clip-a",
    );
    assert.equal(await page.locator("#item-title").textContent(), "Final film");
    assert.deepEqual(await page.locator("#action option").allTextContents(), [
      "Edit video",
      "Regenerate video",
      "Remove video",
    ]);
    assert.equal(await page.locator("#notes").inputValue(), "");
    assert(await page.locator("#prepare").isDisabled());
    assert.equal(await page.locator("#sequence h3").textContent(), "Clips");
    const workspace = await page.locator("#main-workspace").boundingBox();
    for (const selector of ["#sequence", "main > .generation-prompts"]) {
      const section = await page.locator(selector).boundingBox();
      assert(Math.abs(section.width - workspace.width) < 1);
      assert(Math.abs(section.x - workspace.x) < 1);
    }
    const projectName = await page.evaluate(() => project.name);
    await page.locator("#action").selectOption("regenerate");
    assert.equal(
      await page.locator("#notes").inputValue(),
      `Regenerate the ${projectName} video`,
    );
    await page.locator("#action").selectOption("remove");
    assert.equal(
      await page.locator("#notes").inputValue(),
      `Remove the ${projectName} video`,
    );
    await page.locator("#action").selectOption("edit");
    assert.equal(await page.locator("#notes").inputValue(), "");
    assert(
      (
        await page.locator("#sequence-list > button").first().textContent()
      ).includes("FULL VIDEO"),
    );
    assert.equal(
      await page
        .locator("#sequence-list > button")
        .first()
        .getAttribute("aria-pressed"),
      "true",
    );
    assert(
      !(await page.locator("header").textContent()).includes("/ projects"),
    );
    await page.waitForFunction(
      () => document.querySelector("#stage video")?.readyState >= 1,
    );
    await page.waitForFunction(
      () =>
        document.querySelectorAll(
          "#sequence-list > button:not(.full-video) .clip-thumbnail.ready",
        ).length === 2,
    );
    assert(
      await page
        .locator("#sequence-list video")
        .evaluateAll((videos) =>
          videos.every(
            (video) => video.paused && video.currentTime > 0 && !video.poster,
          ),
        ),
    );
    const player = page.locator("#stage video");
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await page.waitForFunction(
      () => document.querySelector("#stage video").currentTime > 0.15,
    );
    await page.getByRole("button", { name: "Pause", exact: true }).click();
    assert(await player.evaluate((m) => m.paused));
    await page.getByRole("slider", { name: "Playback position" }).fill("500");
    assert(Math.abs((await player.evaluate((m) => m.currentTime)) - 2) < 0.25);
    await page
      .getByRole("combobox", { name: "Playback speed" })
      .selectOption("1.5");
    assert.equal(await player.evaluate((m) => m.playbackRate), 1.5);
    await page.getByRole("button", { name: "Mute", exact: true }).click();
    assert(await player.evaluate((m) => m.muted));
    const mainTime = await player.evaluate((m) => m.currentTime);
    await page
      .getByRole("button", { name: "Review Final film", exact: true })
      .click();
    assert(await page.locator("#media-dialog").isVisible());
    await page.waitForFunction(
      () => document.querySelector("#popup-stage video")?.readyState >= 1,
    );
    await page
      .locator("#media-dialog")
      .getByRole("button", { name: "Play", exact: true })
      .click();
    await page.waitForFunction(
      () => document.querySelector("#popup-stage video").currentTime > 0.15,
    );
    await page.getByRole("button", { name: "Close media preview" }).click();
    assert.equal(await player.evaluate((m) => m.currentTime), mainTime);
    await page.waitForFunction(
      () => !document.querySelector("#popup-stage video"),
    );
    assert.equal(await page.locator("#popup-stage video").count(), 0);
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "desktop-review.png"),
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Review Opening shot", exact: true })
      .click();
    assert.equal(
      await page.locator("#item-title").textContent(),
      "Opening shot",
    );
    await page.locator("#sequence-list > button").first().click();
    assert.equal(await page.locator("#item-title").textContent(), "Final film");
    await page.locator("#sequence-list > button").nth(1).click();
    assert.deepEqual(await page.locator("#action option").allTextContents(), [
      "Edit clip",
      "Regenerate clip",
      "Remove clip",
    ]);
    await page.getByLabel("Action", { exact: true }).selectOption("remove");
    const prompt = `Remove clip 1 from the ${projectName} video`;
    assert.equal(
      await page.getByLabel("Prompt", { exact: true }).inputValue(),
      prompt,
    );
    await page.evaluate(() =>
      Object.defineProperty(navigator, "clipboard", {
        value: {
          writeText: async (text) => {
            window.copiedText = text;
          },
        },
        configurable: true,
      }),
    );
    await page.locator("#prepare").click();
    assert.equal(await page.evaluate(() => window.copiedText), prompt);
    await page.locator("#action").selectOption("regenerate");
    assert.equal(
      await page.locator("#notes").inputValue(),
      `Regenerate clip 1 in the ${projectName} video`,
    );
    await page.locator("#action").selectOption("edit");
    assert.equal(await page.locator("#notes").inputValue(), "");
    await page
      .getByLabel("Prompt", { exact: true })
      .fill("Keep the music continuous across the cut.");
    await page.locator("#prepare").click();
    assert.equal(
      await page.evaluate(() => window.copiedText),
      "Keep the music continuous across the cut.",
    );
    // Force the file:// clipboard fallback, which selects the editable field.
    await page.evaluate(() =>
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
      }),
    );
    await page.locator("#prepare").click();
    assert(
      (await page.locator("#prompt-copy-status").textContent()).length > 0,
    );
    assert(
      await page
        .locator("#notes")
        .evaluate((n) => n.selectionEnd - n.selectionStart === n.value.length),
    );
    await page
      .getByRole("button", { name: "Review Cover image", exact: true })
      .click();
    await page.waitForFunction(
      () => document.querySelector("#popup-stage img")?.naturalWidth > 0,
    );
    await page.setViewportSize({ width: 390, height: 844 });
    assert(
      await page
        .locator("#media-dialog")
        .evaluate((dialog) => dialog.scrollWidth <= dialog.clientWidth),
    );
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "mobile-popup.png"),
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    assert.equal(
      await page.locator("#item-title").textContent(),
      "Opening shot",
    );
    await page.locator("#popup-action").selectOption("edit");
    assert.equal(await page.locator("#popup-notes").inputValue(), "");
    assert(await page.locator("#popup-prepare").isDisabled());
    await page.locator("#popup-notes").fill("Make the cover warmer.");
    await page.locator("#popup-prepare").click();
    assert(
      (await page.locator("#popup-prompt-copy-status").textContent()).length >
        0,
    );
    assert.equal(
      await page.locator("#notes").inputValue(),
      "Keep the music continuous across the cut.",
    );
    await page.keyboard.press("Escape");
    assert(!(await page.locator("#media-dialog").isVisible()));
    await page
      .getByRole("button", { name: "Review Night score", exact: true })
      .click();
    await page.waitForFunction(
      () => document.querySelector("audio")?.readyState >= 1,
    );
    await page
      .locator("#media-dialog")
      .getByRole("button", { name: "Play", exact: true })
      .click();
    await page.waitForFunction(
      () => document.querySelector("audio").currentTime > 0.15,
    );
    await page
      .locator("#media-dialog")
      .getByRole("button", { name: "Pause", exact: true })
      .click();
    await page.getByRole("button", { name: "Close media preview" }).click();
    await page.getByRole("tab", { name: "Clips", exact: true }).click();
    assert.equal(await page.locator("#grid .card:visible").count(), 2);
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "mobile-review.png"),
      fullPage: true,
    });
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    assert(
      await page
        .locator("#stage .player-controls")
        .evaluate(
          (n) =>
            n.getBoundingClientRect().right <=
            n.parentElement.getBoundingClientRect().right + 1,
        ),
    );
    // Resolved inputs take precedence over metadata/old requests, including empty lyrics.
    const exactPrompt =
      "integrated_multimodal_description: [Shot 1] A lantern.\n\n" +
      "overall_soundscape: N/A\n\nnon_diegetic_music: N/A\n" +
      "<script>window.injected=1</script> — 夜 " +
      "long-text-".repeat(100);
    await page.evaluate((prompt) => {
      const item = project.items.find((i) => i.kind === "music");
      item.version = 3;
      item.prompt = "Outdated direction";
      item.request = {
        prompt: "Outdated request",
        lyrics: "Old lyrics",
        negative_prompt: "Old exclusion",
      };
      item.generation = {
        prompts: { prompt, negative_prompt: "No harsh percussion", lyrics: "" },
      };
      item.history = [
        {
          version: 1,
          kind: "music",
          prompt: "Earlier direction",
          request: { lyrics: "[Verse]\nEarlier words" },
        },
        {
          version: 2,
          kind: "music",
          generation: {
            prompts: {
              prompt: "Previous caption",
              negative_prompt: "",
              lyrics: "[Verse]\nSaved exact words",
            },
          },
        },
      ];
      choose(item);
      window.copiedText = null;
      Object.defineProperty(navigator, "clipboard", {
        value: {
          writeText: async (text) => {
            window.copiedText = text;
          },
        },
        configurable: true,
      });
    }, exactPrompt);
    assert.equal(
      await page.getByLabel("Used prompt", { exact: true }).inputValue(),
      exactPrompt,
    );
    assert.equal(
      await page.getByLabel("Used negative prompt").inputValue(),
      "No harsh percussion",
    );
    assert.equal(await page.getByLabel("Used lyrics").count(), 0);
    assert(
      (await page.locator("#prompt-fields").textContent()).includes(
        "No lyrics supplied.",
      ),
    );
    assert(
      (await page.locator("#prompt-source").textContent()).includes(
        "Exact text",
      ),
    );
    assert.equal(await page.locator("#prompt-fields script").count(), 0);
    await page
      .locator("#prompt-fields")
      .getByRole("button", { name: "Copy prompt", exact: true })
      .click();
    assert.equal(await page.evaluate(() => window.copiedText), exactPrompt);
    const currentTitle = await page.locator("#item-title").textContent();
    await page.getByLabel("Prompt version").selectOption("1");
    assert.equal(
      await page.getByLabel("Used prompt", { exact: true }).inputValue(),
      "Previous caption",
    );
    assert.equal(
      await page.getByLabel("Used lyrics").inputValue(),
      "[Verse]\nSaved exact words",
    );
    assert.equal(await page.getByLabel("Used negative prompt").count(), 0);
    assert.equal(await page.locator("#item-title").textContent(), currentTitle);
    assert(
      (await page.locator("#prompt-source").textContent()).includes(
        "current item",
      ),
    );
    await page.getByLabel("Prompt version").selectOption("2");
    assert.equal(
      await page.getByLabel("Used prompt", { exact: true }).inputValue(),
      "Earlier direction",
    );
    assert.equal(
      await page.getByLabel("Used lyrics").inputValue(),
      "[Verse]\nEarlier words",
    );
    assert(
      (await page.locator("#prompt-source").textContent()).includes(
        "Resolved generation text was not saved",
      ),
    );
    await page.evaluate(() => {
      Object.defineProperty(navigator, "clipboard", {
        value: undefined,
        configurable: true,
      });
      document.execCommand = () => false;
    });
    await page
      .getByRole("button", { name: "Copy lyrics", exact: true })
      .click();
    assert(
      (await page.locator("#generation-copy-status").textContent()).includes(
        "Press Ctrl+C",
      ),
    );
    assert(
      await page
        .getByLabel("Used lyrics")
        .evaluate((n) => n.selectionEnd - n.selectionStart === n.value.length),
    );
    await page.getByLabel("Prompt version").selectOption("0");
    assert(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    );
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "mobile-prompts.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1100 });
    await page.screenshot({
      path: path.join(path.dirname(process.argv[3]), "desktop-prompts.png"),
      fullPage: true,
    });
    await page.evaluate(() => choose(null));
    assert.equal(await page.locator("#prompt-fields textarea").count(), 0);
    assert(await page.getByLabel("Prompt version").isDisabled());
    // Malicious metadata is text, never executable markup.
    await page.evaluate(() => {
      project.items[0].title = '<img src=x onerror="window.injected=1">';
      choose(project.items[0]);
    });
    assert.equal(await page.locator("#item-title img").count(), 0);
    assert.equal(await page.evaluate(() => window.injected), undefined);
    await page.evaluate(() => {
      project.items.find((i) => i.id === "clip-a").start = 1;
      // Manifest changes normally arrive on page refresh, which clears cached cards.
      sequenceCards.clear();
      document.querySelector("#sequence-list").replaceChildren();
      choose(project.items.find((i) => i.id === "film"));
    });
    await page.waitForFunction(() => {
      const frame = document.querySelector(
        '#sequence-list [data-item-id="clip-a"] .clip-thumbnail.ready video',
      );
      return frame && frame.currentTime >= 1;
    });
    await page.locator("#sequence-list > button").nth(1).click();
    await page.waitForFunction(
      () => document.querySelector("#stage video")?.currentTime >= 1,
    );
    await page.evaluate(() => {
      const original = project.items.find((i) => i.id === "film");
      original.media_revision = 20;
      const later = { ...original, id: "later-film", media_revision: 21 };
      project.items.push(later, {
        id: "later-clip",
        kind: "clip",
        parent: later.id,
      });
      if (latestFullVideo().id !== later.id)
        throw Error("Latest video not selected");
      original.media_revision = 22;
      if (latestFullVideo().id !== original.id)
        throw Error("Updated video not selected");
      original.removed = true;
      if (latestFullVideo().id !== later.id)
        throw Error("Removed video selected");
      later.removed = true;
      if (latestFullVideo() !== null)
        throw Error("Expected no active full video");
    });
    assert.deepEqual(errors, []);
    console.log(
      "PASS: offline previews, controls, prompts/lyrics/history, exact copying, fallback, escaping and mobile layout",
    );
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
