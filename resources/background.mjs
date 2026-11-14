import {createCatalogController, bindCatalogLifecycle} from './live.mjs';
import {ensureOfflineShell, ensureRootOfflineFallback} from './offline.mjs';
import {startAppUpdates} from './app-updates.mjs';

// The same-origin library has its own entry point and must not run twice.
if (typeof document !== 'undefined' && !document.getElementById('resource-grid')) {
  const controller = createCatalogController();
  // Hosted labs keep temporary experiment state until the learner returns home.
  window.appUpdateCanReload = () => !window.location.pathname.startsWith('/labs/');
  startAppUpdates();
  bindCatalogLifecycle(controller);
  void controller.start();
  void ensureOfflineShell().then(shell => shell.available ? ensureRootOfflineFallback() : undefined).catch(() => {});
}
