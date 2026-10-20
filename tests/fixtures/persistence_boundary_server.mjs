// Synthetic loopback boundary only: no Harness profile or provider is loaded.
import { createServer } from 'node:http';
import { registerRuntimeRoutes } from '../../integrations/dsh_ecology_plugin/lib/runtime/routes.js';
import { structuredPhaseError } from '../../integrations/dsh_ecology_plugin/lib/runtime/structured-stage-errors.js';
let route;
registerRuntimeRoutes({ webServer: { register(value) { route=value; return () => {}; } } }, {
  async runStage(body) {
    if (body.stage.startsWith('provider:')) {
      throw structuredPhaseError('model', new Error('private-provider-text'),
        { providerStatus: Number(body.stage.split(':')[1]) });
    }
    if (body.stage === 'untrusted') {
      throw Object.assign(new Error('private-untrusted-text'), {
        code: 'structured_result_persist_host_rejected',
        failure_domain: 'execution_contract',
      });
    }
    if (body.stage === 'schema') throw structuredPhaseError('output_schema');
    throw structuredPhaseError('persistence', Object.assign(new Error('private-sidecar-text'),
      { name:'SidecarError', code:body.stage }));
  },
}, {runtimeToken:'synthetic-loopback-token'});
const server=createServer(route.handler);
server.listen(0,'127.0.0.1',()=>console.log(JSON.stringify({port:server.address().port})));
