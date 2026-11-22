# Model prompting guides

Slopfab embeds an offline catalog of model-specific guidance, templates, original
examples, local capability limits and primary-source links. It needs no model
weights, GPU, network connection or adjacent resource files. Add `--agent` for
structured responses; each successful retrieval returns one `result` with `data`.

```sh
slopfab prompts list
slopfab prompts list --model h3
slopfab prompts get h3 fl2va
slopfab prompts get h3 remake
slopfab prompts select --model h3 --first-frame opening.png --last-frame ending.png --duration 5.875
slopfab prompts select --model h3 --last-frame ending.png
slopfab prompts select --model h3 --reference-image character.png
slopfab prompts select --model h3 --character-swap --reference-image character.png --reference-video source.y4m
slopfab prompts select --model h3 --remake swap --reference-video source.y4m --reference-image painted.png --driving-audio source.wav
slopfab prompts select --model moss --layout
slopfab prompts select --model fibo2 --reference-image room.png --reference-image object.png
slopfab prompts select --model minimax_music3 --instrumental
```

`list` is a compact index. `get MODEL USE_CASE` retrieves one guide, including
the shared model guidance and its sources. Use-case names are case-insensitive;
model names and configured aliases match normal CLI names. `select` chooses from
declared intent and returns the same guide plus ordered `reference_bindings`.
It does not open the supplied paths, inspect images, validate checkpoints or run
generation. A bare `prompts` lists the catalog; `help prompts` describes selectors.

| Model | Use cases |
| --- | --- |
| `h3` | `t2va`, `i2va`, `fl2va`, `l2va`, `ref2va`, `remake`, `character-swap`, `continuation`, `image`, `edit` |
| `krea2` | `image` |
| `moss` | `image`, `layout` |
| `fibo2` | `image`, `edit`, `multi-reference` |
| `yue2` | `song`, `instrumental` |
| `minimax_music3` | `song`, `instrumental` |

## Agent workflow

1. Establish the creative brief and whether prompt review is required. Reuse the
   user's known choices. Use a project for generation and revisions.
2. Retrieve the model/use-case guide. Inspect actual reference media, identify its
   role, and fill the template for the user's brief. Examples illustrate syntax;
   they do not describe the user's files. Fill every remaining `{{placeholder}}`.
3. Validate the generation request. Reconcile resolution, actual sampling-window
   duration and checkpoint/conditioning limits with the prompt. Obtain any
   required prompt approval before generation.
4. Submit the final prompt through the usual request `prompt`, `--prompt-file`
   or `--prompt` input. Keep music `lyrics` separate. Slopfab does not silently
   rewrite prompts or apply templates inside `generate`.

The generated `AGENTS.md`, startup help and `capabilities` advertise the catalog.
Agent launches upgrade older `AGENTS.md` versions from the embedded template;
equal or newer versions stay untouched. Guides are also available through normal
help and capabilities.

## H3 frame roles and local execution

For Viggle Remake, retrieve `prompts get h3 remake`, or select with
`--model h3 --remake swap|edit`, one `--reference-video` and one to four
`--reference-image` paths. Optional `--driving-audio` binds the source soundtrack
as target audio, without introducing an `<Audio N>` reference. Selection returns
ordered source/painted-frame bindings and a mode-specific illustrative example.

The embedded Remake guide summarizes the [official prompting guide](https://huggingface.co/Viggle/Viggle-Remake/blob/main/PROMPTING.md),
with six ordered sections, subject labels, target-description wording, explicit
retention and changes, shot timestamps, painted-frame selection and timeline pins.
It includes original swap/edit examples and maps upstream options to Slopfab.
Remake-specific model guidance replaces generic H3 guidance for this use case;
its FL2VA adapter and text-only conditioning paths differ from ordinary Ref2VA.
Use `--pin` and `--keyframe-until` on the generation command as described by the
guide. Retrieval works offline from the executable; no generation or prompt
rewriting occurs. Fill the remaining shot-timestamp placeholders after validating
the output duration.

The H3 video templates follow the upstream mode-specific alignment prelude and
section order. T2VA has no image alignment. I2VA starts from Picture 1; FL2VA
maps Picture 1 to the opening and Picture 2 to the ending; L2VA maps its **single
Picture 1** to the ending. Ref2VA uses its six-section reference schema.

| Intended input | Selector | Guide | Local generation binding |
| --- | --- | --- | --- |
| Text only | No references | T2VA | Text prompt |
| Opening image | `--first-frame FIRST` | I2VA | `--first-frame FIRST`, compatible FL2VA or Ref2VA weights |
| Ending image | `--last-frame LAST` | L2VA | `--end-anchor LAST`, compatible FL2VA or Ref2VA weights |
| Both endpoints | Both selectors | FL2VA | `--first-frame FIRST --last-frame LAST`, compatible FL2VA or Ref2VA weights |
| Identity/style/content | Ordinary reference selectors | Ref2VA | Ordered ordinary references, compatible Ref2VA weights |
| Saved latent continuation | `--continuation` | Continuation | Consult `help generate` for continuation, Keystone/Capstone and locking |

`--first-frame` and `--last-frame` are both guide selectors and generation
options. They supply fixed temporal anchors on the target canvas, separately
or together. FL2VA, including pi-H3, supports these anchors; compatible Ref2VA
weights also accept them. Supply the video VAE encoder for image conditioning.
Anchors guide the endpoints without guaranteeing exact pixel equality.
FL2VA checkpoints still reject ordinary reference-image inputs.

Generic `--reference-image` inputs never silently become first frames. For a
mixed identity/style reference and endpoint request, load `h3 ref2va` and the
appropriate endpoint guide, then explicitly map all reference roles in native
conditioning order. Automatic selection rejects such mixtures rather than
emitting incorrect Picture numbers. Do not attach the ending image twice.

`--duration` on `prompts select` fills the alignment timestamp to two decimal
places, using a decimal point. Supply the **validated sampling-window duration**;
on continuation this differs from cumulative output length and newly added
duration. Omit it to retain `{{duration}}` for later replacement. `--shots` defaults
to one and fills the ending shot number; it does not invent cuts or a shot list.
Write any additional shots and their cut times yourself. The catalog's examples
retain their own stated assumptions even when selectors customize the template.

ClipProj is text-only in Slopfab; ordinary H3 references need the image-capable
conditioner. Remake's text-label conditioning is a supported exception. A guide
cannot add unsupported conditioning to a model.

## Images and music

Krea guidance covers visual briefs and exploration using the local text-only
path. MOSS adds typography and layout JSON, with centered pixel bounding boxes
matched to the generation canvas. Fibo guidance distinguishes zero-reference
generation, edits and ordered multi-image composition; structured captions use
Slopfab's `edit_instruction` wrapper. Guides describe local limits such as
Fibo glyph conditioning requiring Windows.

Music guides keep the style caption and lyrics separate, with explicit
instrumental variants. Missing lyrics never automatically selects instrumental
mode. YuE2 guidance covers arrangement and planning choices. MiniMax Music3
templates use Global Metadata, Vocal Details and Arrangement. Neither guide
advertises unsupported local audio-reference or cover-song workflows.

## Sources and maintenance

The catalog is [resources/prompting.json](../resources/prompting.json), embedded
in the CLI during CMake configuration. It contains authored summaries and
examples, not a vendored copy of the upstream skills. Each guide includes source
links and the catalog's review date. The DLL and generation algorithms do not
depend on this resource.

- H3: [base video modes](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)
  and [reference modes](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt).
- Viggle Remake: [official prompting guide](https://huggingface.co/Viggle/Viggle-Remake/blob/main/PROMPTING.md)
  and [reference implementation](https://huggingface.co/Viggle/Viggle-Remake/blob/main/inference/sample.py).
- Krea: [official walkthrough](https://www.krea.ai/blog/krea-2-deep-dive-walkthrough)
  and [model repository](https://github.com/krea-ai/krea-2).
- MOSS: [model card](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
  and [user guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md).
- Fibo: [Edit Turbo model card](https://huggingface.co/briaai/Fibo-2.0-edit-turbo)
  and [pipeline documentation](https://huggingface.co/docs/diffusers/main/api/pipelines/bria_fibo2).
- YuE2: [model card](https://huggingface.co/m-a-p/YuE2-3B).
- MiniMax Music3: [model card](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
  and [caption format](https://github.com/MiniMax-AI/MiniMax-Music3/blob/main/skills/music-caption-rewriter/SKILL.md).
