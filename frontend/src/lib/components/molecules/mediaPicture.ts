export interface Box {
  width: number;
  height: number;
}

/**
 * Return the box a picture of the natural size *natural* takes inside *player*, fitted whole: a picture that already fits is not
 * enlarged unless *upscale* says it may be. A size that is not known yet (0) answers an empty box.
 */
export function shownPicture(player: Box, natural: Box, upscale: boolean): Box {
  if (!(player.width > 0 && player.height > 0 && natural.width > 0 && natural.height > 0)) return { width: 0, height: 0 };
  const fit = Math.min(player.width / natural.width, player.height / natural.height);
  const scale = upscale ? fit : Math.min(1, fit);
  return { width: natural.width * scale, height: natural.height * scale };
}
