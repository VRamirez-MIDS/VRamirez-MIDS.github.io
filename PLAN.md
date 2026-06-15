# Victor Ramirez Portfolio — Modernization Plan

---

## Target Audience & Positioning

**Who is looking at this site:** Technical hiring managers and forward-deployed teams at
AI-native companies — Anthropic, Perplexity, Thinking Machines — and DevOps/AI hybrid
companies like JFrog. These are not recruiters; they are engineers who will read the page
with a critical eye.

**Roles being targeted:**
- Forward Deployed AI Architect (JFrog)
- Forward Deployed Engineer / Applied AI (Anthropic, Perplexity)
- Technical Specialist, Claude Code (Anthropic)
- Software Engineer, Developer Productivity / AI Tools (Thinking Machines)
- Member of Technical Staff, Forward Deployed AI (Perplexity)

**What these roles have in common:** Production AI systems, not demos. Customer-facing
technical delivery at enterprise scale. LLM evaluation rigor. Developer platform depth.
Speed of implementation.

**The narrative the site must tell:**
> "I've spent 17 years building production systems at scale. The last three I've been
> building AI infrastructure — RAG pipelines, LLM evaluation frameworks, agentic workflows —
> for hundreds of engineers. I teach it at Techqueria, talk about it on my podcast, and ship
> it live on Cloudflare Workers. I'm not studying AI. I'm running it in production."

**What to stop leading with:** MIDS coursework projects (flight delay prediction, NFL EDA,
movie revenue regression). These read as academic to an Anthropic or Perplexity hiring manager.
They belong in a "Graduate Research" subsection, not the top of the page.

**What to lead with:** Production AI systems (Moody's RAG, LLM eval, agentic workflows),
live demos (AI-Vic chatbot, edge-ai-agent-lab MCP worker), and quantified developer platform
impact.

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

| Asset | Repo | Status |
|---|---|---|
| **Podcast site** (`aibuilderslatinx.com`) | `vhr1975/ai-builders-latinx-edition` | Live — 27 deployments, Git CI active |
| **Gateway Worker** (`edge-ai-agent-lab`) | `ramirez-ai-labs/edge-ai-agent-lab` | Live — Workers AI binding active, 8 deploys |
| **MCP Worker** (`edge-ai-agent-lab-mcp`) | `ramirez-ai-labs/edge-ai-agent-lab` | Live — MCP tools: `time_now`, `worker_info`, `echo` |

- No account setup needed — CF account, Wrangler, and Workers AI binding pattern are all proven
- The gateway worker and MCP worker are **live demos to link from the Projects page**
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
- **Grounding**: System prompt stuffed with resume + bio + projects content (~3-5k tokens, fits Llama 3.1's 128k context — RAG not needed)
- **Chat UI**: React island component (DaisyUI chat bubbles) in Astro

---

## Pages

| Route | Page | Status |
|---|---|---|
| `/` | Home — bio, "What I Build," featured production AI work, writing | Needs restructure |
| `/projects` | Projects — Production AI Systems first, Graduate Research second, Live Demos | Needs restructure |
| `/workshops` | Workshops & Tutorials | Done — placeholder images, some placeholder URLs |
| `/podcast` | AI Builders: LatinX Edition | 70% — Spotify/Apple URLs real, Amazon Music missing |
| `/cv` | Full CV with timeline | Done — needs impact quantification |

---

## Sprint 0: Astro Scaffold ✅ COMPLETE

Branch `astrofy-port` created and pushed. All 5 pages drafted, em-dashes removed,
sidebar and nav wired. PR open against `main`.

---

## Sprint 1: Cloudflare Foundation & Deploy

**Goal:** Get the Astro site live on Cloudflare Pages as a second Pages project in the existing account.

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
- [ ] Fix DMARC record on `aibuilderslatinx.com`
- [ ] Enable Bot Fight Mode on `aibuilderslatinx.com`

> **Old site archiving:** Keep the old GitHub Pages HTML site live throughout Sprint 1 and 2.
> Archive only in Sprint 5, after the new site is confirmed polished and live.

**Deliverable:** Live portfolio on `pages.dev` or custom domain, auto-deploying from `main`.

---

## Sprint 2: Content Restructure & Polish

**Goal:** Reposition the site for the actual target audience. No placeholder content. No
academic-first ordering. Every page tells the "production AI systems builder" story.

### 2a. Home page restructure (highest priority)

- [ ] Add a **"What I Build"** section directly below the bio — scannable, 4-line summary:
  - RAG pipelines and retrieval systems at enterprise scale
  - LLM evaluation frameworks (Recall@k, Precision@k, MRR, groundedness)
  - Agentic workflows and multi-turn agent orchestration
  - Internal developer platforms that ship hundreds of engineers faster
- [ ] Update home headline from "AI Architect & Director" to something that surfaces
  the FDE/Applied AI signal — e.g., "Production AI Systems · Developer Platforms · LatinX in Tech"
- [ ] Add a **"Speaking"** subsection: Techqueria Tech Summit, Oakland Tech Week, Latino AI Summit
- [ ] Replace "Latest Writing" with 3 hand-picked Medium articles that align with target roles
  (RAG, LLM eval, or developer platform topics — not general DS articles)
- [ ] Update featured projects on Home to show Production AI work, not MIDS coursework

### 2b. Projects page restructure (highest priority)

Reorganize into three sections in this order:

**Section 1 — Production AI Systems** (Moody's work, described without exposing confidential
details — system descriptions, not code)
- [ ] RAG Pipeline Infrastructure — production retrieval system, embedding workflows, vector search
- [ ] LLM Evaluation Framework — Recall@k, Precision@k, MRR, groundedness metrics on Databricks/Spark
- [ ] Agentic Workflow Platform — multi-turn agent orchestration, function calling, tool use in production
- [ ] Developer Platform — internal platform serving N+ engineering teams, CI/CD, deployment tooling

**Section 2 — Live Public Demos** (actual running code, linkable)
- [ ] AI-Vic Chatbot — Workers AI + Llama 3.1, system prompt grounding, edge deployment (link once Sprint 3 ships)
- [ ] edge-ai-agent-lab MCP Worker — live MCP implementation: `time_now`, `worker_info`, `echo` tools
  Link: `github.com/ramirez-ai-labs/edge-ai-agent-lab`

**Section 3 — Graduate Research** (MIDS projects, demoted — still present but not leading)
- ML System Engineering & MLOps
- Machine Learning at Scale: Flight Delay Prediction
- Machine Learning: Understanding Hate Crime Patterns
- Data Engineering: Location Recommendations with NoSQL
- Data Analysis: NFL Big Data Bowl
- Statistical Analysis: Movie Revenue Regression Study
- Data Visualization: Travel Guide Reimagined
- Capstone: enRoute (`https://ischool.berkeley.edu/projects/2023/enroute`)

### 2c. CV — quantify impact

- [ ] Add concrete numbers to Director role: number of engineering teams served, onboarding time
  reduction, deployment frequency improvements, scale of AI systems
- [ ] Add concrete numbers to AI Architect role: number of models evaluated, volume of data in
  pipelines, teams using the eval framework
- [ ] Reframe Community section to surface the Anthropic signal: "Teaches Claude, RAG, agents,
  and LLM fundamentals to the LatinX tech community via Techqueria workshops"

### 2d. Podcast page

- [ ] Wire real Spotify URL: `https://open.spotify.com/show/4quI3hyXMd6UjBGy13weMK`
- [ ] Wire real Apple Podcasts URL: `https://podcasts.apple.com/podcast/id1837592967`
- [ ] Add Amazon Music URL when available (currently missing)
- [ ] Add "Recent Episodes" section: 3-5 episodes with title, date, short description, link
- [ ] Reframe host bio to surface Anthropic signal: "teaches Claude, RAG, and agentic systems
  to the LatinX engineering community"

### 2e. Polish

- [ ] Replace all 14 placeholder images with real screenshots or GitHub social preview cards
- [ ] Add Open Graph meta tags to all pages (for LinkedIn/Slack unfurls when sharing with recruiters)
- [ ] Add a custom `favicon.svg` with initials
- [ ] Polish sidebar profile photo — ensure circular mask renders cleanly
- [ ] Wire Techqueria repo URLs: GenAI Tutorial (`github.com/techqueria/GenAI-Tutorial`),
  AI Para Todos (`github.com/techqueria/ai-para-todos`)
- [ ] Re-enable `sitemap` integration
- [ ] Note: `movie-anlysis.jpg` has a typo in the filename — rename or match exactly

**Deliverable:** Site tells a coherent "production AI systems builder" story. Any Anthropic,
JFrog, or Perplexity hiring manager can scan the home page in 10 seconds and see the signal.

---

## Sprint 3: AI-Vic Backend (Cloudflare Worker + Workers AI)

**Goal:** Build the chatbot API. This is also a **live demo artifact** — it demonstrates
Workers AI integration, system prompt grounding, and edge AI deployment to anyone who views
the source or asks about the stack.

> **Model: Cloudflare Workers AI / Llama 3.1 8B Instruct — free tier. NOT the Claude API.**
>
> Reference implementation: `edge-ai-agent-lab` gateway worker (Workers AI binding proven).

### Setup

- [ ] Create new Worker project: `wrangler init ai-vic-worker`
- [ ] Enable Workers AI binding in `wrangler.toml`:

  ```toml
  [ai]
  binding = "AI"
  ```

- [ ] Wire GitHub repo → Workers CI (fix the manual-deploy gap flagged in account audit)

### Grounding Content
- [ ] Extract resume PDF to text (`pdfplumber`)
- [ ] Compile grounding context: CV, all projects (including Production AI section), workshops, podcast
- [ ] Write system prompt — persona rules:
  - Answer in first person as Victor
  - Only answer from provided context
  - Redirect unknown questions gracefully
  - Never invent details

### Worker Implementation
- [ ] `POST /chat` endpoint: accept `{ message, history[] }`, call Llama 3.1, return `{ reply }`
- [ ] CORS headers for Cloudflare Pages domain
- [ ] Basic rate limiting (protect 10k neurons/day free allowance)
- [ ] Deploy: `wrangler deploy`
- [ ] Test with 20+ real visitor questions including role-specific ones:
  - "What's your LLM evaluation methodology?"
  - "Have you deployed RAG in production?"
  - "What's your experience with MCP?"
  - "Tell me about your developer platform work at Moody's"

**Deliverable:** Live Worker endpoint. The AI-Vic chatbot is now a working portfolio piece
that itself demonstrates edge AI deployment skills.

---

## Sprint 4: AI-Vic Frontend (Chat Widget in Astro)

**Goal:** Embed the chatbot on every page as a live, interactive demo of Victor's AI skills.

- [ ] Install React: `npx astro add react`
- [ ] Create `src/components/ChatWidget.tsx` as a React island
- [ ] DaisyUI chat bubble UI:
  - Floating button fixed to bottom-right
  - Expandable panel with toggle
  - User and AI-Vic message bubbles
  - Typing indicator
  - Intro message: "Hi, I'm AI-Vic. Ask me about Victor's work, projects, or background."
- [ ] Wire to Worker endpoint
- [ ] Handle edge cases: empty input, API errors, long responses
- [ ] `<ChatWidget client:load />` in `BaseLayout.astro`
- [ ] Mobile audit

**Deliverable:** Chat widget live on every page. Hiring manager visits site, asks the chatbot
"what's Victor's LLM evaluation experience?" and gets a grounded, accurate answer.

---

## Sprint 5: Launch QA & Go Live

**Goal:** Production-ready. Ship it.

- [ ] End-to-end test all navigation, links, page routes
- [ ] Chatbot QA: role-specific questions (RAG, LLM eval, agentic workflows, developer platform,
  MCP, forward deployed scenarios)
- [ ] Verify chatbot declines to invent information not in context
- [ ] Mobile layout audit
- [ ] Lighthouse audit — target 90+ Performance, Accessibility, SEO
- [ ] Configure `www` redirect and verify HTTPS
- [ ] Final copy review: every page leads with impact, no hedging language
- [ ] **Archive old GitHub Pages site** — only after new site is confirmed live and polished

**Deliverable:** Production portfolio live, AI-Vic chatbot running, fully on Cloudflare free tier.

---

## Content Inventory

### Section 1 — Production AI Systems (to build in Sprint 2)

These are Moody's systems described at an appropriate level — no proprietary code, system-level
descriptions only. Need Victor to confirm what can be described publicly.

| System | Core signal | Status |
|---|---|---|
| RAG Pipeline Infrastructure | Production retrieval, embedding workflows, vector search | To write |
| LLM Evaluation Framework | Recall@k, Precision@k, MRR, groundedness — Databricks/Spark | To write |
| Agentic Workflow Platform | Multi-turn agents, function calling, tool use in prod | To write |
| Developer Platform | Internal IDP, CI/CD, N+ teams, measurable onboarding improvement | To write — need numbers |

### Section 2 — Live Public Demos

| Demo | URL | Status |
|---|---|---|
| AI-Vic Chatbot | `ai-vic-worker.[account].workers.dev` | Sprint 3/4 |
| edge-ai-agent-lab MCP Worker | `github.com/ramirez-ai-labs/edge-ai-agent-lab` | Live now |

### Section 3 — Graduate Research (MIDS)

| Project | Real URL |
|---|---|
| ML System Engineering & MLOps | `https://github.com/VRamirez-MIDS` |
| Machine Learning at Scale: Flight Delay Prediction | `https://github.com/VRamirez-MIDS` |
| Machine Learning: Understanding Hate Crime Patterns | `https://github.com/VRamirez-MIDS` |
| Data Engineering: Location Recommendations with NoSQL | `https://github.com/VRamirez-MIDS` |
| Data Analysis: NFL Big Data Bowl | `https://github.com/VRamirez-MIDS` |
| Statistical Analysis: Movie Revenue Regression Study | `https://github.com/VRamirez-MIDS` |
| Data Visualization: Travel Guide Reimagined | `https://github.com/VRamirez-MIDS` |
| Capstone: enRoute, Running Route Safety App | `https://ischool.berkeley.edu/projects/2023/enroute` |

### Workshops

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

---

## Open Items — Decisions Needed from Victor

- [ ] **CV impact numbers**: What can be stated publicly? e.g., number of engineering teams
  served by the platform, onboarding time improvement %, deployment frequency delta
- [ ] **Production AI system descriptions**: What aspects of Moody's RAG/eval/agentic work
  can be described at a system level without exposing confidential details?
- [ ] **Custom domain**: `victorramirez.dev` or keep `vramirez-mids.github.io`?
- [ ] **Amazon Music podcast URL**: Not yet available — confirm when to add
- [ ] **Podcast episode list**: 3-5 episodes for "Recent Episodes" section (title, date, URL)
- [ ] **Medium articles**: 3 articles for "Latest Writing" on Home — pick ones covering
  RAG, LLM eval, or developer platform topics (strongest signal for target roles)
- [ ] **Project images**: 14 placeholder images to replace — screenshots, GH preview cards, or app images
- [ ] **Chatbot persona name**: Confirm "AI-Vic" or alternative
