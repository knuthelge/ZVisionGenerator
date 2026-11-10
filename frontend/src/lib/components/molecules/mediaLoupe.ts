import { coverTile, type TileCover, type VideoSprite } from '$lib/state/videoSprite.svelte';

/** The loupe's picture is this long on its long side, and at least this long on its short side. */
export const LOUPE_LONG_SIDE = 168;
export const LOUPE_MIN_SHORT_SIDE = 56;
/** The loupe stays this far from the sides of the player. */
export const LOUPE_EDGE = 94;

/** A tile of the contact sheet, placed as the background of a box: the box's size, the sheet's scaled size and the offset. */
export interface LoupeImage extends TileCover {
  width: number;
  height: number;
}

/** Return the box and background placement that show *tile* of *sprite* at the loupe's size: the box is the scaled tile, so the cover is exact. */
export function loupeImage(sprite: VideoSprite, tile: number): LoupeImage {
  const { tileWidth, tileHeight } = sprite;
  const scale = Math.max(LOUPE_LONG_SIDE / Math.max(tileWidth, tileHeight), LOUPE_MIN_SHORT_SIDE / Math.min(tileWidth, tileHeight));
  const box = { width: tileWidth * scale, height: tileHeight * scale };
  return { ...box, ...coverTile(sprite, tile, box) };
}

/** Return the x, from the player's left edge, the loupe is centred on: the pointer, kept `LOUPE_EDGE` from both sides (centred when the player is narrower than that). */
export function loupeCentre(x: number, playerWidth: number): number {
  if (playerWidth <= LOUPE_EDGE * 2) return playerWidth / 2;
  return Math.min(Math.max(x, LOUPE_EDGE), playerWidth - LOUPE_EDGE);
}
