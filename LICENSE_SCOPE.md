# License and material scope

**Author:** Francisco Angulo de Lafuente · **ORCID:** [0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063) · **Copyright:** (c) 2025 Francisco Angulo de Lafuente · **Attribution record date:** 2026-11-06

## Repository license

No root LICENSE file or GitHub-recognized SPDX license was present when this record was made. package.json also has no license field. This attribution pass adds no LICENSE, SPDX identifier, or permission grant. Public repository visibility and authorship metadata do not establish a software license.

## Repository materials

The checked history contains eight commits authored as Francisco Angulo de Lafuente. This supports the repository-level attribution recorded in the code headers and CITATION.cff; it does not establish rights in every linked or externally hosted material.

The tracked README references images hosted as GitHub user attachments; those image files are not in this Git tree. It also links to papers and other repositories. Those materials retain their own provenance and terms and are not licensed by this record.

The project depends on React, Vite, Google Gemini SDK, Recharts, and Heroicons through package.json. Their source code is not vendored here, and this record does not assign their licenses. The platform-specific metadata.json is preserved unchanged.

## Presentation and generated report content

The README is an investor/technical portfolio presentation containing quantitative, ranking, publication, and market claims about projects beyond this application. This repository does not include the underlying experiment logs, competition submissions, or datasets needed to independently validate every claim. The Gemini prompt embeds research summaries and explicitly asks the model to extrapolate descriptions; generated reports may include unsupported statements and need primary-source review.

The Zenodo link in the README identifies a separate paper, “A Physics-Inspired SU(2) Interferometric Pipeline for Anomalous Gravitational-Wave Detection” (DOI 10.5281/zenodo.17285010). It is not a software DOI or license statement for this repository. No software release, tag, or software DOI was identified in this repository metadata.

## Software validation status

During this attribution pass, npm run build completed but transformed only the static HTML entry; index.html has no module script that loads index.tsx. npx tsc --noEmit reports a JSX parse error in App.tsx at the unchanged report-rendering markup. package.json defines no test script. This record does not claim that the app runtime was validated.

## API key handling

vite.config.ts defines process.env.API_KEY from GEMINI_API_KEY for client-side code. If the React app is wired into a build, Vite will substitute that value into browser JavaScript. Do not deploy a bundle built with a real API key; route Gemini requests through a server-side proxy first. The tracked repository contained no API key during this audit.
