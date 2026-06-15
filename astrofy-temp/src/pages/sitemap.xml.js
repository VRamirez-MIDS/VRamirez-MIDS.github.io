const SITE = 'https://vramirez-mids.github.io';

const pages = ['', 'projects', 'workshops', 'podcast', 'cv'];

export async function GET() {
  const urls = pages
    .map(
      (page) => `
  <url>
    <loc>${SITE}/${page}</loc>
    <changefreq>monthly</changefreq>
  </url>`
    )
    .join('');

  return new Response(
    `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${urls}
</urlset>`,
    { headers: { 'Content-Type': 'application/xml' } }
  );
}
