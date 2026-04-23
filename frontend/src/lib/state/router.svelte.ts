import type { PageId, RouterState } from '$lib/types';

// Parses both hash ('#/workspace?...') and query-string ('?page=workspace&...')
// Hash takes precedence over query string
function parseLocation(): RouterState {
  const hash = window.location.hash;
  const query = window.location.search;

  // Try hash first: #/page?params or #/page
  const hashMatch = hash.match(/^#\/([a-z]+)(\?.*)?$/);
  if (hashMatch) {
    const page = hashMatch[1] as PageId;
    const params: Record<string, string> = {};
    if (hashMatch[2]) {
      new URLSearchParams(hashMatch[2].slice(1)).forEach((v, k) => { params[k] = v; });
    }
    return { page: isValidPage(page) ? page : 'workspace', params };
  }

  // Fall back to query string: ?page=workspace&...
  const searchParams = new URLSearchParams(query);
  const page = (searchParams.get('page') ?? 'workspace') as PageId;
  const params: Record<string, string> = {};
  searchParams.forEach((v, k) => { if (k !== 'page') params[k] = v; });
  return { page: isValidPage(page) ? page : 'workspace', params };
}

function isValidPage(page: string): page is PageId {
  return ['workspace', 'gallery', 'config', 'models'].includes(page);
}

// Reactive router state (Svelte 5 rune)
let _router = $state<RouterState>(parseLocation());

export const router = {
  get page(): PageId { return _router.page; },
  get params(): Record<string, string> { return _router.params; },

  navigate(page: PageId, params: Record<string, string> = {}): void {
    const hashParams = new URLSearchParams(params).toString();
    const hash = `#/${page}${hashParams ? '?' + hashParams : ''}`;
    window.history.pushState({}, '', hash);
    _router = { page, params };
  },

  replace(page: PageId, params: Record<string, string> = {}): void {
    const hashParams = new URLSearchParams(params).toString();
    const hash = `#/${page}${hashParams ? '?' + hashParams : ''}`;
    window.history.replaceState({}, '', hash);
    _router = { page, params };
  }
};

// Listen for browser back/forward
if (typeof window !== 'undefined') {
  window.addEventListener('popstate', () => {
    _router = parseLocation();
  });
}
