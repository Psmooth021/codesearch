// Minimal request router for the demo web layer.

export function matchRoute(routes, path) {
  return routes.find((route) => route.pattern.test(path));
}

export class Router {
  constructor() {
    this.routes = [];
  }

  register(pattern, handler) {
    this.routes.push({ pattern, handler });
  }

  dispatch(path) {
    const match = matchRoute(this.routes, path);
    if (!match) {
      throw new Error(`no route for ${path}`);
    }
    return match.handler(path);
  }
}
