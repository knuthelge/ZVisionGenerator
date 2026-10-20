import { describe, expect, it } from 'vitest';
import { claimsSheetPull, sheetDragOutcome } from './sheetDrag';

describe('sheetDragOutcome', () => {
  it('closes after a long pull down', () => {
    expect(sheetDragOutcome({ offset: 160, velocity: 0.1, height: 400 })).toBe('close');
  });

  it('springs back after a short slow pull', () => {
    expect(sheetDragOutcome({ offset: 40, velocity: 0.1, height: 400 })).toBe('stay');
  });

  it('closes after a flick, judged by how fast the finger moved as it lifted', () => {
    expect(sheetDragOutcome({ offset: 30, velocity: 0.9, height: 400 })).toBe('close');
  });

  it('needs a quarter of a tall sheet', () => {
    expect(sheetDragOutcome({ offset: 120, velocity: 0, height: 800 })).toBe('stay');
    expect(sheetDragOutcome({ offset: 200, velocity: 0, height: 800 })).toBe('close');
  });

  it('stays when a long pull ends with a flick back up', () => {
    expect(sheetDragOutcome({ offset: 200, velocity: -1, height: 400 })).toBe('stay');
  });

  it('never closes on a pull up', () => {
    expect(sheetDragOutcome({ offset: -200, velocity: -2, height: 400 })).toBe('stay');
  });
});

describe('claimsSheetPull', () => {
  function sheet(scrolled: boolean): { root: HTMLElement; handle: HTMLElement; body: HTMLElement; field: HTMLElement } {
    const root = document.createElement('div');
    root.innerHTML = '<div data-sheet-handle><span class="title">Title</span></div><div class="body"><p class="row">Row</p><input></div>';
    const body = root.querySelector('.body') as HTMLElement;
    Object.defineProperty(body, 'scrollHeight', { value: 900 });
    Object.defineProperty(body, 'clientHeight', { value: 300 });
    Object.defineProperty(body, 'scrollTop', { value: scrolled ? 120 : 0 });
    return { root, handle: root.querySelector('.title') as HTMLElement, body: root.querySelector('.row') as HTMLElement, field: root.querySelector('input') as HTMLElement };
  }

  it('always pulls from the handle, up or down, even with the content scrolled', () => {
    const { root, handle } = sheet(true);
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: 3, target: handle }, root)).toBe(true);
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: -3, target: handle }, root)).toBe(true);
  });

  it('pulls from the body down only while its content is at the top', () => {
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: 2, target: sheet(false).body }, sheet(false).root)).toBe(true);
    const scrolled = sheet(true);
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: 2, target: scrolled.body }, scrolled.root)).toBe(false);
  });

  it('leaves a push up, a sideways slide and a field to the content', () => {
    const { root, body, field } = sheet(false);
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: -2, target: body }, root)).toBe(false);
    expect(claimsSheetPull({ axis: 'x', dx: 5, dy: 1, target: body }, root)).toBe(false);
    expect(claimsSheetPull({ axis: 'y', dx: 0, dy: 4, target: field }, root)).toBe(false);
  });
});
