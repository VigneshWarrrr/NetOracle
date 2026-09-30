# NetOracle brand assets — "Midnight Oracle"

The product name and wordmark is always **NetOracle** (one word, capital N and O).

## Mark

N/O network-orbit monogram. It combines:

- an orbit ring (the O), with one orbiting node,
- an N drawn through it,
- two network nodes at the free ends of the N.

The gap in the ring is cut with an SVG mask rather than painted in a background colour, so every mark works on any background and in a single flat colour.

## Files

| File | Use | Colours |
|---|---|---|
| `netoracle-logo-light.svg` | Horizontal logo (mark + wordmark) for **light** backgrounds: sidebar, header | ink `#0A0A0A`, gold `#EAB308` nodes with ink outline |
| `netoracle-logo-dark.svg` | Horizontal logo for **dark** backgrounds | white `#F8FAFC` N and wordmark, gold `#FACC15` ring and nodes |
| `netoracle-logo-stacked-light.svg` | Stacked logo (mark above wordmark) for light backgrounds: landing, marketing | as logo-light |
| `netoracle-logo-stacked-dark.svg` | Stacked logo for dark backgrounds | as logo-dark |
| `netoracle-mark-light.svg` | Monogram only, for light backgrounds: sidebar, mobile, app icon | as logo-light |
| `netoracle-mark-dark.svg` | Monogram only, for dark backgrounds | as logo-dark |
| `netoracle-mark.svg` | Flat one-colour monogram (`currentColor`; black when used as `<img>`) | single colour |
| `favicon.svg` | Simplified 32×32 mark on a midnight tile; legible at 16 px on light or dark browser chrome | `#0A0A0A` tile, gold ring and node, white N |

`-light` / `-dark` refer to the **background** the asset is designed for.

## Rules

- Keep clear space of at least 25% of the mark's height on every side. Don't recolour, stretch, add effects or glows, or re-type the wordmark.
- Gold is the **brand** accent (logo, brand moments). It is not a risk, warning or severity colour.
- On light surfaces, gold is never used as text: `#EAB308` on white is about 1.9:1. Use `--net-brand-text` (`#A16207`) for gold-toned text.
- The wordmark is live SVG text with the `Inter` → `Segoe UI` → `Helvetica Neue` → `Arial` fallbacks and a fixed `textLength`. Its width is stable, but the glyphs follow whichever of those fonts is installed. Convert it to outlines when a font-independent master is needed, for example for print.

## Tokens

The colour, type and shape tokens are in `static/css/netoracle-tokens.css` (`--net-*`). It is imported at the top of `styles.css`.
