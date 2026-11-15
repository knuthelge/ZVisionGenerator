import { vi } from 'vitest';

/**
 * A stand-in WebGL2 context for jsdom, which has none: every method is a spy (`getExtension` yields an object whose `loseContext` is one too), every constant a number, and shader and
 * program builds report success. `calls(name)` returns the arguments of each call of a method.
 */
export function fakeGl(): { gl: WebGL2RenderingContext; calls: (name: string) => unknown[][] } {
  const spies = new Map<string, ReturnType<typeof vi.fn>>();
  const spyOf = (name: string): ReturnType<typeof vi.fn> => {
    if (!spies.has(name)) spies.set(name, vi.fn(() => ({})));
    return spies.get(name)!;
  };
  const gl = new Proxy(
    {},
    {
      get(_target, name: string) {
        if (name === 'getShaderParameter' || name === 'getProgramParameter') return () => true;
        if (name === 'isContextLost') return () => false;
        if (/^[A-Z0-9_]+$/.test(name)) return name.length;
        if (name === 'getExtension') return () => ({ loseContext: spyOf('loseContext') });
        return spyOf(name);
      },
    },
  );
  return { gl: gl as unknown as WebGL2RenderingContext, calls: (name) => spies.get(name)?.mock.calls ?? [] };
}
