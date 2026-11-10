import { browseSpriteUrl } from '$lib/api/browse';

/** Tiles across, and rows down, of a contact sheet. The same grid as `SPRITE_COLUMNS` in `utils/video_sprite.py`. */
export const SPRITE_COLUMNS = 11;

/** The most frames a sheet holds. */
export const SPRITE_MAX_TILES = SPRITE_COLUMNS * SPRITE_COLUMNS;

/** A loaded contact sheet: where to fetch the picture and how big one tile is in it. */
export interface VideoSprite {
  url: string;
  tileWidth: number;
  tileHeight: number;
}

/** Return how many tiles a sheet of a clip of *frames* frames holds. */
export function spriteTiles(frames: number): number {
  return Math.min(frames, SPRITE_MAX_TILES);
}

/** Return the tile that shows *frame* of a clip of *frames* frames: `round(frame / (frames - 1) * (tiles - 1))`. A one-frame clip is tile 0. */
export function spriteTile(frame: number, frames: number): number {
  const lastFrame = frames - 1;
  const tiles = spriteTiles(frames);
  if (lastFrame <= 0 || tiles <= 1) return 0;
  const clamped = Math.min(Math.max(frame, 0), lastFrame);
  return Math.round((clamped / lastFrame) * (tiles - 1));
}

/** Return the column and row of *tile* in the sheet's grid. */
export function spriteCell(tile: number): { column: number; row: number } {
  return { column: tile % SPRITE_COLUMNS, row: Math.floor(tile / SPRITE_COLUMNS) };
}

/** The size of the box a tile of the contact sheet fills. */
export interface TileBox {
  width: number;
  height: number;
}

/** The background that shows one tile of a sheet, cropped to fill a box: the sheet's scaled size and its offset in the box's pixels. */
export interface TileCover {
  sheetWidth: number;
  sheetHeight: number;
  offsetX: number;
  offsetY: number;
}

/** Return the background placement that shows *tile* of *sprite* filling *box*, scaled to cover it and centred, as `object-fit: cover` does. */
export function coverTile(sprite: VideoSprite, tile: number, box: TileBox): TileCover {
  const scale = Math.max(box.width / sprite.tileWidth, box.height / sprite.tileHeight);
  const tileWidth = sprite.tileWidth * scale;
  const tileHeight = sprite.tileHeight * scale;
  const { column, row } = spriteCell(tile);
  return {
    sheetWidth: tileWidth * SPRITE_COLUMNS,
    sheetHeight: tileHeight * SPRITE_COLUMNS,
    offsetX: -column * tileWidth + (box.width - tileWidth) / 2,
    offsetY: -row * tileHeight + (box.height - tileHeight) / 2,
  };
}

// The contact sheets of the videos seen this session. The server builds a sheet the first time it is asked for, so a player
// asks on the first hover and the Extend bar on its first show, once per path. A sheet that failed to load is forgotten,
// so a later view may ask again (each view asks once, not on every hover); everything that draws from it has a fallback meanwhile.
class VideoSpriteStore {
  #loaded = $state.raw<Record<string, VideoSprite>>({});
  #wanted = new Set<string>();

  /** The sheet of the video at *path*: `null` until it has loaded, and while its sheet cannot be built. */
  of(path: string): VideoSprite | null {
    return this.#loaded[path] ?? null;
  }

  /** Fetch the sheet of the video at *path*; once per path. */
  ensure(path: string): void {
    if (this.#wanted.has(path)) return;
    this.#wanted.add(path);
    const url = browseSpriteUrl('source_video', path);
    const image = new Image();
    image.onload = () => {
      this.#loaded = { ...this.#loaded, [path]: { url, tileWidth: image.naturalWidth / SPRITE_COLUMNS, tileHeight: image.naturalHeight / SPRITE_COLUMNS } };
    };
    image.onerror = () => {
      this.#wanted.delete(path);
    };
    image.src = url;
  }

  /** Forget every sheet. For tests. */
  reset(): void {
    this.#loaded = {};
    this.#wanted.clear();
  }
}

export const videoSprite = new VideoSpriteStore();
