# Victor Ramirez Portfolio — Modernization Plan

## Goal
Replace the current static HTML site with a modern Astro + Tailwind portfolio hosted entirely
on Cloudflare's free tier, including an AI-Vic chatbot grounded in Victor's resume and content.

---

## Architecture

```
GitHub (source) → Cloudflare Pages (Astro site, free)
                       ↕
              Cloudflare Worker (AI-Vic API, free)
                       ↕
              Cloudflare Workers AI / Llama 3.1 (free)
```

### Cloudflare Free Tier Usage

| Service | Free Allowance | Estimated Usage |
|---|---|---|
| Cloudflare Pages | 500 builds/mo, unlimited traffic | Well within |
| Cloudflare Workers | 100k requests/day | Fine for portfolio traffic |
| Workers AI | 10k neurons/day | ~50-100 chat messages/day |

---

## Tech Stack

- **Framework**: Astro 4 + Tailwind CSS + DaisyUI
- **Template base**: Astrofy
- **Hosting**: Cloudflare Pages
- **Chatbot API**: Cloudflare Worker
- **AI Model**: Cloudflare Workers AI / Llama 3.1 8B Instruct
- **Grounding**: System prompt stuffed with resume + bio + projects + workshops content
- **Chat UI**: React island component (DaisyUI chat bubbles) in Astro

---

## Pages

| Route | Page | Status |
|---|---|---|
| `/` | Home (bio, featured projects, writing) | Draft |
| `/projects` | 8 projects in 3 categories | Draft |
| `/workshops` | 6 workshops in 2 categories | Draft |
| `/podcast` | AI Builders: LatinX Edition | Draft |
| `/cv` | Full CV with timeline | Draft |

---

## Sprint 1: Cloudflare Foundation & Deploy

**Goal:** Get the current site live on Cloudflare Pages, replace GitHub Pages.

- [ ] Create Cloudflare account and connect GitHub repo to Cloudflare Pages
- [ ] Configure build settings (`npm run build`, output: `dist/`, root: `astrofy-temp/`)
- [ ] Verify auto-deploy on every push to `main`
- [ ] Remove unused template pages: `store/`, `services/`, demo blog posts in `src/content/blog/`
- [ ] Remove Blog from navigation, replace with link to Medium
- [ ] Verify `output: 'static'` in `astro.config.mjs` for Cloudflare Pages compatibility
- [ ] Set up custom domain on Cloudflare Pages (or use `.pages.dev` subdomain)
- [ ] Archive / redirect old GitHub Pages site

**Deliverable:** Live site on `pages.dev` or custom domain, replacing the old HTML site.

---

## Sprint 2: Site Polish

**Goal:** Turn the draft into a finished product. No placeholder content anywhere.

### Visual & Content
- [ ] Replace all Astro logo placeholder images with real project screenshots or GitHub preview cards
- [ ] Add real Spotify / Apple Podcasts / Amazon Music URLs to the Podcast page
- [ ] Replace "Latest Writing" section on Home with 3 hand-picked Medium article cards (title, description, link to Medium) — no blog system needed
- [ ] Add Open Graph meta tags to all pages (title, description, image) for LinkedIn/Slack unfurls
- [ ] Add a custom `favicon.svg` with initials or personal mark
- [ ] Add a "Speaking" subsection to Home page: Techqueria Tech Summit, Oakland Tech Week, Latino AI Summit
- [ ] Polish sidebar profile photo — ensure circular mask renders cleanly
- [ ] Review all copy: lead with impact, no hedging language, no generic bullets

### Technical
- [ ] Remove `services.astro` and `store/` pages entirely
- [ ] Fix RSS feed warning: rename `get` export to `GET` in `rss.xml.js`
- [ ] Re-enable `sitemap` integration now that site URL is configured
- [ ] Verify `robots.txt` has correct site URL

**Deliverable:** Polished, complete portfolio ready for professional use.

---

## Sprint 3: AI-Vic Backend (Cloudflare Worker + Workers AI)

**Goal:** Build and deploy the API that powers the chatbot.

### Setup
- [ ] Install Wrangler CLI: `npm install -g wrangler`
- [ ] Create Worker project: `wrangler init ai-vic-worker`
- [ ] Enable Workers AI binding in `wrangler.toml`:
  ```toml
  [ai]
  binding = "AI"
  ```

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
- [ ] Add basic rate limiting to protect free tier
- [ ] Deploy: `wrangler deploy`
- [ ] Test via curl / Postman with 20+ real visitor questions

**Deliverable:** Live Worker at `ai-vic.yourname.workers.dev/chat` returning accurate answers about Victor.

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
- [ ] Confirm old GitHub Pages site is archived or redirected

**Deliverable:** Production site live at custom domain with AI-Vic chatbot, fully on Cloudflare free tier.

---

## Content Inventory

### Projects (8 total)
1. ML System Engineering & MLOps — Kubernetes, Docker, FastAPI, MLflow
2. Machine Learning at Scale: Flight Delay Prediction — Spark, Hadoop, MapReduce
3. Machine Learning: Understanding Hate Crime Patterns — TensorFlow, linear regression
4. Data Engineering: Location Recommendations with NoSQL — Neo4j, MongoDB, Redis
5. Data Analysis: NFL Big Data Bowl — Python, NumPy, Pandas
6. Statistical Analysis: Movie Revenue Regression Study — OLS regression
7. Data Visualization: Travel Guide Reimagined — Tableau
8. Capstone: enRoute, Running Route Safety App — iOS, ML

### Workshops (6 total)
1. GenAI Tutorial: Chatbots, Memory & RAG Systems
2. OpenAI API Tutorials: GPT Models, Fine-Tuning & Integration
3. AI Mastery Hub: Beginner to Expert Learning Path
4. AI Para Todos: Accessible AI Workshop Series
5. Software Architecture Showcase
6. AI vs ML vs DL: The Definitive Guide

### Community & Speaking
- Podcast: AI Builders: LatinX Edition
- Teaching: Techqueria (RAG, embeddings, agents, LLM fundamentals)
- Speaking: Techqueria Tech Summit, Oakland Tech Week, Latino AI Summit
- Writing: medium.com/@vhr1975

---

## Open Items / Decisions Needed

- [ ] Custom domain? (e.g. `victorramirez.dev` or keep `vramirez-mids.github.io`)
- [ ] Real Spotify / Apple Podcasts URLs for the podcast page
- [ ] Project screenshot images to replace Astro placeholders
- [ ] Medium article URLs for the "Latest Writing" section (pick 3 best)
- [ ] Confirm chatbot persona name: "AI-Vic" or something else
