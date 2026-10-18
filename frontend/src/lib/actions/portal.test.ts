import { describe, expect, it } from 'vitest';
import { portal } from './portal';

describe('portal', () => {
  it('moves the node to the body and removes it on destroy', () => {
    const host = document.createElement('div');
    document.body.appendChild(host);
    const node = document.createElement('div');
    host.appendChild(node);

    const action = portal(node);
    expect(node.parentElement).toBe(document.body);
    expect(host.contains(node)).toBe(false);

    action.destroy();
    expect(node.isConnected).toBe(false);
    host.remove();
  });
});
