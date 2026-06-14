# Victor Ramirez Portfolio — Modernization Plan

## Goal
Replace the current static HTML site with a modern Astro + Tailwind portfolio hosted entirely
on Cloudflare's free tier, including an AI-Vic chatbot grounded in Victor's resume and content.

---

## Architecture

```
GitHub (source) → Cloudflare Pages (Astro site, free)
                       ↕
              Cloudflare Worker — AI-Vic (new, free)
                       ↕
              Cloudflare Workers AI / Llama 3.1 8B Instruct (free, NOT Claude API)
```

### Existing Cloudflare Account — AI Builders Studio: LatinX

The CF account is already active with proven infrastructure. The portfolio site will be a
second Pages project added to the same account.

| Asset | Repo | Status |
|---|---|---|
| **Podcast site** (`aibuilderslatinx.com`) | `vhr1975/ai-builders-latinx-edition` | Live — 27 deployments, Git CI active |
| **Gateway Worker** (`edge-ai-agent-lab`) | `ramirez-ai-labs/edge-ai-agent-lab` | Live — Workers AI binding active, 8 deploys |
| **MCP Worker** (`edge-ai-agent-lab-mcp`) | `ramirez-ai-labs/edge-ai-agent-lab` | Live — MCP tools: `time_now`, `worker_info`, `echo` |

**What this means for the portfolio build:**

- No account setup needed — CF account, Wrangler, and Workers AI binding pattern are all proven
- Pages CI (GitHub → Cloudflare Pages) is working and tested on the podcast site — same pattern applies here
- The gateway worker is the reference implementation for the AI-Vic worker (Workers AI binding, worker-to-worker fetch pattern)
- Domain `aibuilderslatinx.com` belongs to the podcast — portfolio needs its own domain or uses `.pages.dev`

### Cloudflare Free Tier Usage

| Service | Free Allowance | Estimated Usage |
|---|---|---|
| Cloudflare Pages | 500 builds/mo, unlimited traffic | Well within (second Pages project) |
| Cloudflare Workers | 100k requests/day | Fine for portfolio traffic |
| Workers AI | 10k neurons/day | ~50-100 chat messages/day |

### Known Account Security Gaps (fix alongside Sprint 1)

| Gap | Severity | Fix |
|---|---|---|
| DMARC record error on `aibuilderslatinx.com` | Moderate | Add/correct DMARC TXT in DNS |
| Bot Fight Mode not enabled | Low | One-toggle in Security settings |
| AI crawl controls unconfigured | Low | Decide intentionally, then toggle |
| Both Workers have no Git CI — Wrangler manual only | Operational risk | Wire to CI in Sprint 3 |

---

## Tech Stack

- **Framework**: Astro 4 + Tailwind CSS + DaisyUI
- **Template base**: Astrofy
- **Package manager**: npm (not pnpm — `package-lock.json` is authoritative, `pnpm-lock.yaml` ignored)
- **Hosting**: Cloudflare Pages
- **Chatbot API**: Cloudflare Worker
- **AI Model**: Cloudflare Workers AI / Llama 3.1 8B Instruct (free tier — NOT the Claude API)
- **Grounding**: System prompt stuffed with resume + bio + projects + workshops content (~3-5k tokens, fits Llama 3.1's 128k context — RAG not needed)
- **Chat UI**: React island component (DaisyUI chat bubbles) in Astro

---

## Pages

| Route | Page | Status |
|---|---|---|
| `/` | Home (bio, featured projects, writing) | 80% — blog section still pulls dummy posts |
| `/projects` | 8 projects in 3 categories | Done — placeholder images, some placeholder URLs |
| `/workshops` | 6 workshops in 2 categories | Done — placeholder images, some placeholder URLs |
| `/podcast` | AI Builders: LatinX Edition | 70% — host bio real, Spotify/Apple URLs real, Amazon Music URL missing |
| `/cv` | Full CV with timeline | Done |

---

## Sprint 0: Astro Scaffold ✅ COMPLETE

Branch `astrofy-port` created and pushed. All 5 pages drafted, em-dashes removed,
sidebar and nav wired. PR open against `main`.

---

## Sprint 1: Cloudflare Foundation & Deploy

**Goal:** Get the Astro site live on Cloudflare Pages as a second Pages project in the existing account.

> CF account, Wrangler, and Pages CI are already proven — no account setup needed. Mirror the
> podcast site pattern: connect repo → set build root → push to deploy.

- [ ] Add portfolio as a new Pages project in the existing CF account
  - Repo: `VRamirez-MIDS/VRamirez-MIDS.github.io`, branch: `main`
  - Build command: `npm run build` (not pnpm)
  - Build output: `dist/`
  - Root directory: `astrofy-temp/`
- [ ] Verify auto-deploy fires on every push to `main`
- [ ] Remove unused template pages: `store/`, `services/`, demo blog posts in `src/content/blog/`
- [ ] Remove Blog/Store/Services from navigation, replace Blog link with Medium
- [ ] Verify `output: 'static'` in `astro.config.mjs` for Pages compatibility
- [ ] Fix RSS feed export: rename `get` → `GET` in `rss.xml.js`
- [ ] Decide and configure custom domain (or use `.pages.dev` for now)
- [ ] Fix DMARC record on `aibuilderslatinx.com` (account-level gap, low effort)
- [ ] Enable Bot Fight Mode on `aibuilderslatinx.com` (one toggle)

> **Old site archiving:** Do NOT archive or redirect the old GitHub Pages HTML site during
> this sprint. Keep it live until the new Astro site is fully polished and ready. Archiving
> happens in Sprint 5 only.

**Deliverable:** Live portfolio on `pages.dev` or custom domain, auto-deploying from `main`.

---

## Sprint 2: Site Polish

**Goal:** Turn the draft into a finished product. No placeholder content anywhere.

### Real URLs to wire in

| Location | Field | Real URL |
|---|---|---|
| Podcast page | Spotify | `https://open.spotify.com/show/4quI3hyXMd6UjBGy13weMK` |
| Podcast page | Apple Podcasts | `https://podcasts.apple.com/podcast/id1837592967` |
| Podcast page | Amazon Music | Not yet available — use `https://music.amazon.com/podcasts` as placeholder |
| Projects page | enRoute capstone | `https://ischool.berkeley.edu/projects/2023/enroute` |
| Workshops page | GenAI Tutorial | `https://github.com/techqueria/GenAI-Tutorial` |
| Workshops page | AI Para Todos | `https://github.com/techqueria/ai-para-todos` |

### Images (14 placeholders to replace)

All cards currently use `/post_img.webp` (Astrofy default). Replace with real screenshots,
GitHub social preview cards, or app store images. Note: if using `movie-anlysis.jpg`, the
filename has a typo — rename it or match exactly in the `img` prop.

### Visual & Content
- [ ] Replace all 14 placeholder images with real project/workshop screenshots or GitHub preview cards
- [ ] Wire real Spotify and Apple Podcasts URLs (see table above)
- [ ] Add "Recent Episodes" section to podcast page (3-5 episodes: title, date, short description, link)
- [ ] Replace "Latest Writing" section on Home with 3 hand-picked Medium article cards (title, desc, link) — no blog system needed
- [ ] Add Open Graph meta tags to all pages (title, description, image) for LinkedIn/Slack unfurls
- [ ] Add a custom `favicon.svg` with initials or personal mark
- [ ] Add a "Speaking" subsection to Home page: Techqueria Tech Summit, Oakland Tech Week, Latino AI Summit
- [ ] Polish sidebar profile photo — ensure circular mask renders cleanly
- [ ] Review all copy: lead with impact, no hedging language, no generic bullets

### Technical
- [ ] Remove `services.astro` and `store/` pages entirely
- [ ] Re-enable `sitemap` integration now that site URL is configured
- [ ] Verify `robots.txt` has correct site URL

**Deliverable:** Polished, complete portfolio ready for professional use.

---

## Sprint 3: AI-Vic Backend (Cloudflare Worker + Workers AI)

**Goal:** Build and deploy the API that powers the chatbot.

> **Model: Cloudflare Workers AI / Llama 3.1 8B Instruct — free tier. This is NOT the Claude API.**
> The Claude API is opt-in only and has no role in this stack.
>
> Wrangler is already installed and in use. Workers AI binding pattern is proven in the
> `edge-ai-agent-lab` gateway worker — use it as the reference implementation.

### Setup

- [ ] Create new Worker project: `wrangler init ai-vic-worker`
- [ ] Enable Workers AI binding in `wrangler.toml` (same pattern as `edge-ai-agent-lab`):

  ```toml
  [ai]
  binding = "AI"
  ```

- [ ] Wire GitHub repo → Workers CI to avoid manual Wrangler deploys (gap flagged in account audit)

### Grounding Content
- [ ] Extract resume PDF to text via `pdfplumber`
- [ ] Compile grounding context from:
  - CV page (roles, education, skills)
  - Projects page (all 8 projects with descriptions)
  - Workshops page (all 6 workshops)
  - Podcast page
  - Resume PDF text
- [ ] Write system prompt with persona rules:
  - Answer in first person as Victor
  - Only answer from provided context
  - Redirect unknown questions gracefully
  - Never invent details

### Worker Implementation
- [ ] Implement `POST /chat` endpoint:
  - Accept `{ message, history[] }`
  - Call `env.AI.run("@cf/meta/llama-3.1-8b-instruct", { messages: [...] })`
  - Return `{ reply }`
- [ ] Add CORS headers for Cloudflare Pages domain
- [ ] Add basic rate limiting to protect free tier (10k neurons/day)
- [ ] Deploy: `wrangler deploy`
- [ ] Test via curl / Postman with 20+ real visitor questions

**Deliverable:** Live Worker at `ai-vic-worker.[account].workers.dev/chat` returning accurate answers about Victor.

---

## Sprint 4: AI-Vic Frontend (Chat Widget in Astro)

**Goal:** Embed the chatbot into the site on every page.

- [ ] Install React integration: `npx astro add react`
- [ ] Create `src/components/ChatWidget.tsx` as a React island
- [ ] Build widget UI with DaisyUI chat bubble components:
  - Floating button fixed to bottom-right corner
  - Expandable chat panel (toggle open/close)
  - User and AI-Vic message bubbles
  - Typing indicator while awaiting response
  - "Ask me anything about Victor..." placeholder text
  - Intro message: "Hi, I'm AI-Vic. Ask me about Victor's work, projects, or background."
- [ ] Wire widget to Worker endpoint
- [ ] Handle edge cases: empty input, API errors, long responses
- [ ] Add `<ChatWidget client:load />` to `BaseLayout.astro` so it appears on every page
- [ ] Test on mobile (chat panel usable on small screens)

**Deliverable:** Chat widget live on every page, grounded in Victor's resume.

---

## Sprint 5: Launch QA & Go Live

**Goal:** Production-ready, no rough edges.

- [ ] End-to-end test all navigation, links, and page routes
- [ ] Chatbot QA: test 20+ real visitor questions across all content areas
- [ ] Verify chatbot declines to invent information not in the context
- [ ] Mobile layout audit (sidebar collapses, chat widget usable on phone)
- [ ] Lighthouse audit — target 90+ on Performance, Accessibility, SEO
- [ ] Configure `www` redirect and verify HTTPS on custom domain
- [ ] Final copy review across all pages
- [ ] **Archive old GitHub Pages site** — only once the new Astro site is confirmed live and
  fully polished. Options: remove old HTML files from repo root, redirect via `_redirects`,
  or move to an `archive/` branch.

**Deliverable:** Production site live at custom domain with AI-Vic chatbot, fully on Cloudflare free tier.

---

## Content Inventory

### Projects (8 total)

| Project | Real URL |
|---|---|
| ML System Engineering & MLOps | `https://github.com/VRamirez-MIDS` (no dedicated repo yet) |
| Machine Learning at Scale: Flight Delay Prediction | `https://github.com/VRamirez-MIDS` |
| Machine Learning: Understanding Hate Crime Patterns | `https://github.com/VRamirez-MIDS` |
| Data Engineering: Location Recommendations with NoSQL | `https://github.com/VRamirez-MIDS` |
| Data Analysis: NFL Big Data Bowl | `https://github.com/VRamirez-MIDS` |
| Statistical Analysis: Movie Revenue Regression Study | `https://github.com/VRamirez-MIDS` |
| Data Visualization: Travel Guide Reimagined | `https://github.com/VRamirez-MIDS` |
| Capstone: enRoute, Running Route Safety App | `https://ischool.berkeley.edu/projects/2023/enroute` |

### Workshops (6 total)

| Workshop | Real URL |
|---|---|
| GenAI Tutorial: Chatbots, Memory & RAG Systems | `https://github.com/techqueria/GenAI-Tutorial` |
| OpenAI API Tutorials | `https://github.com/VRamirez-MIDS` |
| AI Mastery Hub | `https://github.com/VRamirez-MIDS` |
| AI Para Todos | `https://github.com/techqueria/ai-para-todos` |
| Software Architecture Showcase | `https://github.com/VRamirez-MIDS` |
| AI vs ML vs DL | `https://github.com/VRamirez-MIDS` |

### Podcast

| Platform | URL |
|---|---|
| Spotify | `https://open.spotify.com/show/4quI3hyXMd6UjBGy13weMK` |
| Apple Podcasts | `https://podcasts.apple.com/podcast/id1837592967` |
| Amazon Music | Not yet available |

### Community & Speaking
- Podcast: AI Builders: LatinX Edition
- Teaching: Techqueria (RAG, embeddings, agents, LLM fundamentals)
- Speaking: Techqueria Tech Summit, Oakland Tech Week, Latino AI Summit
- Writing: medium.com/@vhr1975

---

## Open Items / Decisions Needed

- [ ] Custom domain? (e.g. `victorramirez.dev` or keep `vramirez-mids.github.io`)
- [ ] Amazon Music podcast URL (not yet available — check back)
- [ ] Individual GitHub repo URLs for MIDS projects (or confirm org-level links are acceptable)
- [ ] Project screenshot images to replace 14 Astro placeholders
- [ ] Medium article URLs for the "Latest Writing" section (pick 3 best)
- [ ] Confirm chatbot persona name: "AI-Vic" or something else
- [ ] Podcast episode list for Sprint 2 (titles, dates, episode URLs for "Recent Episodes" section)
