import type { ModelLicence } from '$lib/types';

/** A link to a licence's text or its model page. */
export interface LicenceLink {
  label: string;
  href: string;
  ariaLabel: string;
}

/** The link of a licence record. It opens in a new tab. */
export function licenceLinks(licence: ModelLicence): LicenceLink[] {
  return [{ label: `${licence.url_label} ↗`, href: licence.url, ariaLabel: `${licence.name} ${licence.url_label.toLowerCase()} (opens in a new tab)` }];
}
