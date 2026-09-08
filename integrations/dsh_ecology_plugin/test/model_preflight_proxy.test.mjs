import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { Readable, Writable } from "node:stream";
import { registerApiProxy, MODEL_PREFLIGHT_PROXY_TIMEOUT_MS } from "../lib/web/proxy.js";

class Response extends Writable {
  constructor() { super(); this.statusCode = null; this.headersSent = false; this.chunks = []; }
  _write(chunk, _encoding, callback) { this.chunks.push(Buffer.from(chunk)); callback(); }
  writeHead(status) { this.statusCode = status; this.headersSent = true; return this; }
  json() { return JSON.parse(Buffer.concat(this.chunks).toString() || "{}"); }
}
function request(path, method = "POST") {
  const req = Readable.from(method === "POST" ? [Buffer.from("{}")] : []);
  req.url = path; req.method = method; req.headers = {"content-type":"application/json"};
  return req;
}

test("only canonical POST model-preflight waits beyond ordinary proxy timeout", async () => {
  const seen=[];
  const upstream=createServer((req,res)=>{
    seen.push({method:req.method,url:req.url,authorization:req.headers.authorization});
    req.resume();
    const timer=setTimeout(()=>{
      const body=JSON.stringify({passed:true,scope:"tool_and_schema_transport_only"});
      res.writeHead(200,{"content-type":"application/json","content-length":Buffer.byteLength(body)});res.end(body);
    },100);
    res.once("close",()=>clearTimeout(timer));
  });
  await new Promise(resolve=>upstream.listen(0,"127.0.0.1",resolve));
  let proxy;
  registerApiProxy({webServer:{register(value){proxy=value;}}},{backendOrigin:new URL(`http://127.0.0.1:${upstream.address().port}`),serviceToken:"service-test-only",totalTimeoutMs:20,maxBodyBytes:10000,maxResponseBytes:10000});
  try {
    const success=new Response();await proxy.handler(request("/api/ecology-evolution/model-preflight"),success);
    assert.equal(success.statusCode,200);assert.equal(success.json().passed,true);
    assert.equal(MODEL_PREFLIGHT_PROXY_TIMEOUT_MS,250000);
    assert.equal(seen[0].authorization,"Bearer service-test-only");
    for(const [path,method] of [["/api/ecology-evolution/health","GET"],["/api/ecology-evolution/runs","POST"],["/api/ecology-evolution/model-preflight","GET"],["/api/ecology-evolution/model-preflight/extra","POST"],["/api/ecology-evolution/%6dodel-preflight","POST"]]){
      const res=new Response();await proxy.handler(request(path,method),res);assert.equal(res.statusCode,502,`${method} ${path}`);
      assert.doesNotMatch(JSON.stringify(res.json()),/service-test-only/);
    }
  } finally { upstream.closeAllConnections();await new Promise(resolve=>upstream.close(resolve)); }
});
