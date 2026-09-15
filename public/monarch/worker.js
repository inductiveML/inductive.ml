var J9=["timestamp-query","shader-f16","subgroups"];function Z9(L){return J9.includes(L)}function o1(L){if(!Z9(L))throw Error(`${L} is not a registered requestable feature`);return L}function J7(){return{dispatches:0,dispatchesIndirect:0,submits:0,commandBuffers:0,passes:0,bindGroups:0,writeBuffers:0,writeBufferBytes:0,createBuffers:0,createBufferBytes:0,destroyBuffers:0,pipelines:0,pipelinesAsync:0,pipelineCompileMs:0,shaderModules:0,mapAsyncs:0,copyBufferToBuffer:0,submitOffsets:[],injected:0}}function X9(L){let $={},J=Object.getPrototypeOf(L);for(let Z of Object.getOwnPropertyNames(J)){if(Z==="constructor")continue;let X=Object.getOwnPropertyDescriptor(J,Z);if(X===void 0||typeof X.get!=="function")continue;let Y=X.get.call(L);if(typeof Y==="number")$[Z]=Y}return $}function Y9(L){let $={};if(L===void 0)return $;for(let[J,Z]of Object.entries(L))if(typeof Z==="number")$[J]=Z;return $}(()=>{if(typeof navigator>"u"||navigator.gpu===void 0){globalThis.__probeHooks={installed:!1,configure:()=>{return},beginStep:()=>{return},endStep:()=>{throw Error("WebGPU unavailable")},snapshot:()=>{throw Error("WebGPU unavailable")},setInjection:()=>{return},enableTimestamps:()=>{return},disableTimestamps:()=>{return},collectTimestamps:()=>Promise.resolve([]),devices:()=>[],descriptors:()=>[],errors:()=>[],lost:()=>[],liveBytes:()=>0,suspend:(U)=>U(),suspendAsync:(U)=>U()};return}let L=!1,$=!1,J=0,Z=0,X=J7(),Y=null,Q=null,H=0,j=[],V=[],G=[],I=[],D=[],R=new WeakMap,z=new WeakMap,T=()=>L&&!$,N=GPUAdapter.prototype,y=N.requestDevice;N.requestDevice=async function(U){let A=U===void 0?[]:[...U.requiredFeatures??[]],M=[...A],P=[];for(let v of j){let J1=o1(v);if(!M.includes(J1)&&this.features.has(J1))M.push(J1),P.push(J1)}let S={...U??{},requiredFeatures:M},O=this.info,w={requestedFeatures:A.map(String),requiredFeatures:M.map(String),addedFeatures:P.map(String),requiredLimits:Y9(U?.requiredLimits),label:U?.label??"",adapter:{vendor:O.vendor,architecture:O.architecture,device:O.device,description:O.description,isFallbackAdapter:O.isFallbackAdapter,subgroupMinSize:O.subgroupMinSize,subgroupMaxSize:O.subgroupMaxSize,features:[...this.features].map(String).sort(),limits:X9(this.limits)}},u=await y.call(this,S);return V.push(u),G.push(w),u.addEventListener("uncapturederror",(v)=>{I.push(`${v.error.constructor.name}: ${v.error.message}`)}),u.lost.then((v)=>{D.push(`${v.reason}: ${v.message}`)}),u};let E=GPUDevice.prototype,t=E.createBuffer;E.createBuffer=function(U){let A=t.call(this,U);if(R.set(A,U.size),H+=U.size,T())X.createBuffers+=1,X.createBufferBytes+=U.size;return A};let e=E.createBindGroup;E.createBindGroup=function(U){if(T())X.bindGroups+=1;return e.call(this,U)};let k=E.createShaderModule;E.createShaderModule=function(U){if(T())X.shaderModules+=1;return k.call(this,U)};let h=E.createComputePipeline;E.createComputePipeline=function(U){let A=performance.now(),M=h.call(this,U);if(T())X.pipelines+=1,X.pipelineCompileMs+=performance.now()-A;return M};let L1=E.createComputePipelineAsync;E.createComputePipelineAsync=async function(U){let A=performance.now(),M=await L1.call(this,U);if(T())X.pipelinesAsync+=1,X.pipelineCompileMs+=performance.now()-A;return M};let U1=GPUBuffer.prototype,w1=U1.destroy;U1.destroy=function(){let U=R.get(this);if(U!==void 0)H-=U,R.delete(this);if(T())X.destroyBuffers+=1;return w1.call(this)};let R1=U1.mapAsync;U1.mapAsync=function(U,A,M){if(T())X.mapAsyncs+=1;return R1.call(this,U,A,M)};let B=GPUQueue.prototype,I1=B.submit;B.submit=function(U){let A=[...U];if(T()){if(Y!==null)A=[...A,Y],Y=null,X.injected+=1;X.submits+=1,X.commandBuffers+=A.length,X.submitOffsets.push(performance.now()-J)}return I1.call(this,A)};let i=B.writeBuffer;B.writeBuffer=function(U,A,M,P,S){if(T()){X.writeBuffers+=1;let O=S!==void 0?S:M.byteLength-(P??0);X.writeBufferBytes+=O}return i.call(this,U,A,M,P,S)};let n=GPUCommandEncoder.prototype,$1=n.beginComputePass;n.beginComputePass=function(U){let A=U,M=null;if(T()){if(X.passes+=1,Q!==null&&Q.next<Q.capacity&&(U===void 0||U.timestampWrites===void 0)){let S=Q.next;Q.next+=1,A={...U??{},timestampWrites:{querySet:Q.querySet,beginningOfPassWriteIndex:2*S,endOfPassWriteIndex:2*S+1}},M={index:S,dispatches:0},Q.passes.push(M)}}let P=$1.call(this,A);if(M!==null)z.set(P,M);return P};let j1=n.copyBufferToBuffer,x1=j1,H1=function(U,A,M,P,S){if(T())X.copyBufferToBuffer+=1;if(typeof A==="number"){if(!(M instanceof GPUBuffer)||P===void 0)throw TypeError("copyBufferToBuffer: bad five-argument form");return j1.call(this,U,A,M,P,S)}if(M!==void 0&&typeof M!=="number")throw TypeError("copyBufferToBuffer: bad three-argument form");return x1.call(this,U,A,M)};n.copyBufferToBuffer=H1;let K1=GPUComputePassEncoder.prototype,W1=K1.dispatchWorkgroups;K1.dispatchWorkgroups=function(U,A,M){if(T()){X.dispatches+=1;let P=z.get(this);if(P!==void 0)P.dispatches+=1}return W1.call(this,U,A,M)};let B1=K1.dispatchWorkgroupsIndirect;K1.dispatchWorkgroupsIndirect=function(U,A){if(T()){X.dispatches+=1,X.dispatchesIndirect+=1;let M=z.get(this);if(M!==void 0)M.dispatches+=1}return B1.call(this,U,A)};let G1=()=>{let U=L?performance.now():Z,A=X.submitOffsets[0],M=X.submitOffsets[X.submitOffsets.length-1];return{dispatches:X.dispatches,dispatches_indirect:X.dispatchesIndirect,submits:X.submits,command_buffers:X.commandBuffers,passes:X.passes,bind_groups:X.bindGroups,write_buffers:X.writeBuffers,write_buffer_bytes:X.writeBufferBytes,create_buffers:X.createBuffers,create_buffer_bytes:X.createBufferBytes,destroy_buffers:X.destroyBuffers,pipelines:X.pipelines,pipelines_async:X.pipelinesAsync,pipeline_compile_ms:X.pipelineCompileMs,shader_modules:X.shaderModules,map_asyncs:X.mapAsyncs,copy_buffer_to_buffer:X.copyBufferToBuffer,first_submit_ms:A===void 0?-1:A,last_submit_ms:M===void 0?-1:M,submit_offsets_ms:[...X.submitOffsets],injected:X.injected,live_gpu_bytes:H,step_ms:U-J}},C={installed:!0,configure(U){j=[...U]},beginStep(){if(X=J7(),Q!==null)Q.next=0,Q.passes=[];J=performance.now(),Z=J,L=!0},endStep(){return Z=performance.now(),L=!1,G1()},snapshot:G1,setInjection(U){Y=U},enableTimestamps(U,A){if(Q!==null)C.disableTimestamps();if(A<1||A>2048)throw Error(`timestamp capacity ${A} pairs outside 1..2048`);let M=U.createQuerySet({type:"timestamp",count:2*A}),P=U.createBuffer({size:16*A,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),S=U.createBuffer({size:16*A,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});Q={device:U,querySet:M,capacity:A,next:0,resolveBuffer:P,stagingBuffer:S,passes:[]}},disableTimestamps(){if(Q===null)return;Q.querySet.destroy(),Q.resolveBuffer.destroy(),Q.stagingBuffer.destroy(),Q=null},async collectTimestamps(){let U=Q;if(U===null||U.next===0)return[];let A=$;$=!0;try{let M=U.device.createCommandEncoder();M.resolveQuerySet(U.querySet,0,2*U.next,U.resolveBuffer,0),M.copyBufferToBuffer(U.resolveBuffer,0,U.stagingBuffer,0,16*U.next),U.device.queue.submit([M.finish()]),await U.stagingBuffer.mapAsync(GPUMapMode.READ,0,16*U.next);let P=new BigUint64Array(U.stagingBuffer.getMappedRange(0,16*U.next).slice(0));U.stagingBuffer.unmap();let S=[];for(let O of U.passes){let w=P[2*O.index],u=P[2*O.index+1];if(w===void 0||u===void 0)throw Error("timestamp index out of range");S.push({index:O.index,dispatches:O.dispatches,begin_ns:Number(w),end_ns:Number(u)})}return S}finally{$=A}},devices:()=>V,descriptors:()=>G,errors:()=>I,lost:()=>D,liveBytes:()=>H,suspend(U){let A=$;$=!0;try{return U()}finally{$=A}},async suspendAsync(U){let A=$;$=!0;try{return await U()}finally{$=A}}};globalThis.__probeHooks=C})();var Z7={host:"127.0.0.1",port:47315,transformersBundle:"dist/transformers.js",transformersRoute:"/vendor/transformers/transformers.js",ortRoute:"/vendor/ort/",ortWasmFiles:["ort-wasm-simd-threaded.asyncify.mjs","ort-wasm-simd-threaded.asyncify.wasm"],modelsRoute:"/models/",weightsRoute:"/weights/",pageRoute:"/page.js",hooksRoute:"/hooks.js"};var e1={layers:14,layerTypes:["conv","conv","attention","conv","attention","conv","attention","conv","attention","conv","attention","conv","attention","conv"],hidden:1024,ffn:2560,vocab:65536,heads:16,kvHeads:8,headDim:64,convCache:3,convBias:!1,normEps:0.00001,ropeTheta:1e6,ropeInterleaved:!1,tieEmbedding:!0,quantBits:4,quantBlock:32,maxPositionEmbeddings:128000,bosTokenId:1,eosTokenId:7},s={weightBits:4,residualBytes:4,hiddenBytes:2,kvCacheBytes:2,convCacheBytes:2,logitsBytes:4,accumulate:"f32"},A0={commandBuffersPerToken:1,passesPerToken:1,readbackEveryTokens:32,readbackRingTokens:64,concatQkv:!0,concatGateUp:!0},f={mlpFused:{workgroupSize:256,slices:80},attnSplit:{coreWorkgroupSize:256},flash:{coreWorkgroupSize:64,tilePositions:64,maxBlocks:32,positionsPerBlockRule:64,mergeWorkgroupSize:64,partialWords:68},fold:{workgroupSize:256,threadsPerRow:8},geometry:{base:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:4,subgroupsPerRow:1},wide:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1}}};var X7=[{key:"A2",splitN:!1,flash:!1,dispatches:74},{key:"A2N1",splitN:!0,flash:!1,dispatches:74},{key:"A2N2",splitN:!1,flash:!0,dispatches:80},{key:"A2N1N2",splitN:!0,flash:!0,dispatches:80}];var t1={ropeRows:1024,ropeRowsExtended:2560,headPreScale:2,sectionAlignment:256,weightsFile:"weights.bin",manifestFile:"weights_manifest.json",maxQueries:1},g={matvec:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:4,subgroupsPerRow:1,maxCols:2560,quantBlock:32},matvecWide:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1,maxCols:2560,quantBlock:32},conv:{workgroupSize:256},embed:{workgroupSize:256},sample:{workgroupSize:256,partials:256},maxPositions:2560};var Y7={maxStorageBuffersPerShaderStage:10,maxBindingsPerBindGroup:18,maxBindGroups:1,maxComputeWorkgroupStorageSize:23552,maxComputeWorkgroupStorageSizeFlash:18176,maxComputeInvocationsPerWorkgroup:256,maxComputeWorkgroupSizeX:256,maxComputeWorkgroupsPerDimension:4096};var Q9={chainCounts:[8,16,24,32,48,64,96,128],rounds:8,repsPerRound:25,warmupReps:20,workgroupSize:64,shapes:[{key:"attn_o_proj",mode:"matvec_residual",rows:1024,cols:1024,emits:1024,zeroPoints:!1,preScale:1,weightBytes:655360,dispatchBytes:663552},{key:"mlp_down",mode:"matvec_residual",rows:1024,cols:2560,emits:1024,zeroPoints:!1,preScale:1,weightBytes:1638400,dispatchBytes:1651712},{key:"attn_qkv",mode:"norm_projection",rows:2048,cols:1024,emits:2048,zeroPoints:!1,preScale:1,weightBytes:1310720,dispatchBytes:1327104},{key:"conv_in_proj",mode:"norm_matvec",rows:3072,cols:1024,emits:3072,zeroPoints:!1,preScale:1,weightBytes:1966080,dispatchBytes:1980416},{key:"mlp_gate_up",mode:"norm_swiglu",rows:5120,cols:1024,emits:2560,zeroPoints:!1,preScale:1,weightBytes:3276800,dispatchBytes:3290112},{key:"head",mode:"norm_head",rows:65536,cols:1024,emits:65536,zeroPoints:!0,preScale:t1.headPreScale,weightBytes:42991616,dispatchBytes:43257856}],rowsPerWorkgroup:[1,2,4,8,16,32],rowsPerWorkgroupHead:[16,32,64,128,256,512,1024],workgroupSizes:[64,128,256],reachabilityRule:"subgroupsPerRow * subgroupSize <= cols / quantBlock",dispatchCounts:[4,16],footprintBytes:754974720,fillChunkBytes:67108864,curveRounds:6,curveTargetBytesPerRound:314572800,curveMinReps:3,curveMaxReps:25,curveWarmupReps:8,kneeFraction:0.9,seed:111007};function s1(L,$,J,Z,X){if(!Number.isInteger(L)||L<=0||!Number.isInteger($)||$<=0)throw Error(`quantMatrixBytes: bad shape ${L}x${$}`);if($%X!==0)throw Error(`quantMatrixBytes: cols ${$} not a multiple of block ${X}`);if(Z!==4)throw Error(`quantMatrixBytes: only 4-bit packing is registered, got ${Z}`);let Y=$/X,Q=L*$*Z/8,H=L*Y*4,j=J?L*Math.ceil(Y/2):0;return{quant:Q,scales:H,zeroPoints:j,total:Q+H+j}}function H9(L){if(!Number.isInteger(L)||L<3)throw Error(`layerTypesFor: layers must be an integer >= 3, got ${L}`);let $=[];for(let J=0;J<L;J+=1)$.push(J>=2&&J%2===0?"attention":"conv");return $}function Q7(L,$){let J=L.layerTypes[$];if(J===void 0)throw Error(`layerMatrices: layer ${$} out of range`);let{hidden:Z,ffn:X,kvHeads:Y,headDim:Q,quantBits:H,quantBlock:j}=L,V=Y*Q,G=(D,R,z,T)=>({name:R,layer:$,role:D,rows:z,cols:T,zeroPoints:!1,bytes:s1(z,T,!1,H,j)}),I=[G("mlp.gate_proj",`/model/layers.${$}/mlp/gate_proj/MatMul_Q4`,X,Z),G("mlp.up_proj",`/model/layers.${$}/mlp/up_proj/MatMul_Q4`,X,Z),G("mlp.down_proj",`/model/layers.${$}/mlp/down_proj/MatMul_Q4`,Z,X)];if(J==="conv")return[G("conv.in_proj",`/model/layers.${$}/conv/in_proj/MatMul_Q4`,3*Z,Z),G("conv.out_proj",`/model/layers.${$}/conv/out_proj/MatMul_Q4`,Z,Z),...I];return[G("attn.q_proj",`/model/layers.${$}/attn/q_proj/MatMul_Q4`,Z,Z),G("attn.k_proj",`/model/layers.${$}/attn/k_proj/MatMul_Q4`,V,Z),G("attn.v_proj",`/model/layers.${$}/attn/v_proj/MatMul_Q4`,V,Z),G("attn.o_proj",`/model/layers.${$}/attn/o_proj/MatMul_Q4`,Z,Z),...I]}function H7(L){return{name:"logits",layer:-1,role:"lm_head",rows:L.vocab,cols:L.hidden,zeroPoints:!0,bytes:s1(L.vocab,L.hidden,!0,L.quantBits,L.quantBlock)}}var P1={layerTypes:H9(14),hidden:1024,ffn:2560,vocab:65536,heads:16,kvHeads:8,headDim:64,convCache:3,quantBits:4,quantBlock:32};var j7=["attn_o_proj","mlp_down","attn_qkv","conv_in_proj","mlp_gate_up","head"];var j9={conv_in_proj:"conv_in_proj",conv_out_proj:"attn_o_proj",attn_qkv:"attn_qkv",attn_o_proj:"attn_o_proj",mlp_gate_up:"mlp_gate_up",mlp_down:"mlp_down",head:"head"},K9=["conv_in_proj","conv_out_proj","attn_o_proj","mlp_gate_up","mlp_down"];var x0=8;function O0(L){let{workgroupSize:$,subgroupSize:J,rowsPerSubgroup:Z,subgroupsPerRow:X}=L;if($%J!==0)throw Error(`rowsPerWorkgroup: workgroup ${$} is not a multiple of subgroup ${J}`);let Y=$/J;if(!Number.isInteger(X)||X<1)throw Error(`rowsPerWorkgroup: subgroupsPerRow must be a positive integer, got ${X}`);if(Y%X!==0)throw Error(`rowsPerWorkgroup: ${Y} subgroups do not partition into groups of ${X}`);if(!Number.isInteger(Z)||Z<1)throw Error(`rowsPerWorkgroup: rowsPerSubgroup must be a positive integer, got ${Z}`);return Y/X*Z}function K7(L,$,J){if($%J!==0)throw Error(`cellReachable: ${$} columns are not a multiple of the ${J}-wide block`);return L.subgroupsPerRow*L.subgroupSize<=$/J}function V9(L,$,J){if($.splitN&&K9.includes(L))return J.knee[j9[L]];return J.carried[L]}function V7(L,$,J){if(!Number.isInteger(L)||L<1)throw Error(`blocksForPositions: ${L} is not a positive position count`);if(!Number.isInteger($)||$<1)throw Error(`blocksForPositions: ${$} is not a positive block length`);return Math.min(J,Math.max(1,Math.ceil(L/$)))}function d1(L,$,J){let Z=L[0];if(Z===void 0)throw Error("fuse: empty source list");for(let H of L){if(H.cols!==Z.cols)throw Error(`fuse: column mismatch ${H.name} ${H.cols} != ${Z.cols}`);if(H.zeroPoints!==Z.zeroPoints)throw Error(`fuse: zero-point mismatch in ${H.name}`)}let X=L.reduce((H,j)=>H+j.rows,0),Y=s1(X,Z.cols,Z.zeroPoints,J,$),Q=L.reduce((H,j)=>H+j.bytes.total,0);if(Y.total!==Q)throw Error(`fuse: concatenated bytes ${Y.total} != sum of sources ${Q}`);return{sources:L,rows:X,cols:Z.cols,zeroPoints:Z.zeroPoints,bytes:Y.total}}function v1(L,$){let J=L.find((Z)=>Z.role===$);if(J===void 0)throw Error(`plan: no matrix with role ${$}`);return J}function b1(L,$,J,Z,X,Y){let Q=s1($,J,Z,X.quantBits,X.quantBlock).total;return{section:L,rows:$,cols:J,zeroPoints:Z,bytes:Q,kSlices:Y}}function U7(L,$){return $.splitN?0:L.layerTypes.length}function U9(L,$){return U7(L,$)%2===0}function F0(L,$){let{queries:J,position:Z,precision:X,levers:Y,slices:Q,flash:H,geometry:j}=$;if(!Number.isInteger(J)||J<1)throw Error(`planStep: queries must be a positive integer, got ${J}`);let V=!Y.splitN;if(!U9(L,Y))throw Error(`planStep: ${U7(L,Y)} fused blocks at ${L.layerTypes.length} layers is odd, so the residual ping-pong does not close and the token cannot be encoded once and replayed`);if(V&&J!==1)throw Error(`planStep: the fused MLP is a single-query decode specialisation, got q = ${J}`);if(!Number.isInteger(H.blocks)||H.blocks<1||H.blocks>H.maxBlocks)throw Error(`planStep: ${H.blocks} position blocks is outside 1..${H.maxBlocks}`);let{hidden:G,ffn:I,vocab:D,heads:R,headDim:z,kvHeads:T,convCache:N,quantBlock:y,quantBits:E}=L;if(R%T!==0)throw Error(`planStep: ${R} q heads do not divide into ${T} kv heads`);let t=R/T,e=T*z,k=G*X.residualBytes*J,h=G*X.hiddenBytes*J,L1=G*4,w1=(R*z+2*T*z)*4*J,R1=[],B={residual:0,partials:0,pending:0},I1=(C,U)=>Math.ceil(C/O0(U))*J,i=(C,U,A,M,P,S,O,w,u,v,J1,q1,_1,U0)=>{if(!Number.isInteger(v)||v<1)throw Error(`planStep: ${C} has a non-positive workgroup count ${v}`);if(q1===null!==(_1===null))throw Error(`planStep: ${C} carries a site without a geometry, or the reverse`);R1.push({index:R1.length,name:C,kind:U,layer:A,weight:M,reads:P,constantBytes:S,readBytes:O,writeBytes:w,fusion:u,workgroups:v,workgroupSize:J1,site:q1,geometry:_1,attnPart:U0})},n=(C,U,A,M,P,S,O,w,u,v,J1)=>{let q1=V9(v,Y,j);i(C,U,A,M,P,S,O,w,$1(),I1(u,q1),q1.workgroupSize,v,q1,J1)},$1=()=>({foldSlices:0,slices:0,residualIn:B.residual,residualOut:B.residual,partialsIn:B.partials,partialsOut:B.partials}),j1=(C)=>{if(B.pending===0)return;let U=B.pending;i(C,"fold",-1,null,[],0,k+U*L1,k,{foldSlices:U,slices:0,residualIn:B.residual,residualOut:B.residual,partialsIn:B.partials,partialsOut:B.partials},Math.ceil(G/(j.foldGeometry.workgroupSize/j.foldGeometry.threadsPerRow)),j.foldGeometry.workgroupSize,null,null,"none"),B.pending=0},x1=(C)=>{let U={foldSlices:B.pending,slices:C,residualIn:B.residual,residualOut:1-B.residual,partialsIn:B.partials,partialsOut:1-B.partials};return B.residual=1-B.residual,B.partials=1-B.partials,B.pending=C,U},H1=G/y,K1=G/2+H1*4+Math.ceil(H1/2);i("embed","embed",-1,null,[],K1*J,4*J,k,$1(),J,j.embedWorkgroupSize,null,null,"none");for(let C=0;C<L.layerTypes.length;C+=1){let U=L.layerTypes[C],A=Q7(L,C),M=G*4,P=2*(z/2)*4*J;if(U==="conv"){let w=d1([v1(A,"conv.in_proj")],y,E),u=d1([v1(A,"conv.out_proj")],y,E),v=G*N*X.convCacheBytes;j1(`L${C}.conv.fold`),n(`L${C}.conv.in_proj`,"norm_matvec",C,w,[b1(`L${C}.conv.in_proj`,w.rows,w.cols,!1,L,0)],M,k,3*h,w.rows,"conv_in_proj","none"),i(`L${C}.conv.core`,"conv_core",C,null,[],G*N*4,3*h+v,h+v,$1(),Math.ceil(G/j.convWorkgroupSize)*J,j.convWorkgroupSize,null,null,"none"),n(`L${C}.conv.out_proj`,"matvec_residual",C,u,[b1(`L${C}.conv.out_proj`,u.rows,u.cols,!1,L,0)],0,h+k,k,u.rows,"conv_out_proj","none")}else{let w=d1([v1(A,"attn.q_proj"),v1(A,"attn.k_proj"),v1(A,"attn.v_proj")],y,E),u=d1([v1(A,"attn.o_proj")],y,E),v=Z+J,J1=2*e*X.kvCacheBytes*J;if(j1(`L${C}.attn.fold`),n(`L${C}.attn.qkv`,"attn_proj",C,w,[b1(`L${C}.attn.qkv`,w.rows,w.cols,!1,L,0)],M,k,w1,w.rows,"attn_qkv","projection"),Y.flash){let q1=t*z*4*J,_1=H.blocks*R*H.partialWords*4*J;i(`L${C}.attn.flash`,"attn_core_flash",C,null,[],2*z*4+P,T*H.blocks*q1+T*2*z*4*J+2*T*Z*z*X.kvCacheBytes,_1+J1,$1(),T*H.blocks*J,j.flashCoreWorkgroupSize,null,null,"core"),i(`L${C}.attn.merge`,"attn_merge",C,null,[],0,_1,h,$1(),R*J,j.mergeWorkgroupSize,null,null,"merge")}else i(`L${C}.attn.core`,"attn_core_qkv",C,null,[],2*z*4+P,w1+2*T*v*z*X.kvCacheBytes,h+J1,$1(),T*J,j.attnCoreWorkgroupSize,null,null,"core");n(`L${C}.attn.o_proj`,"matvec_residual",C,u,[b1(`L${C}.attn.o_proj`,u.rows,u.cols,!1,L,0)],0,h+k,k,u.rows,"attn_o_proj","out_projection")}let S=d1([v1(A,"mlp.gate_proj"),v1(A,"mlp.up_proj")],y,E),O=d1([v1(A,"mlp.down_proj")],y,E);if(Y.splitN)j1(`L${C}.mlp.fold`),n(`L${C}.mlp.gate_up`,"norm_matvec_swiglu",C,S,[b1(`L${C}.mlp.gate_up`,S.rows,S.cols,!1,L,0)],M,k,I*X.hiddenBytes*J,I,"mlp_gate_up","none"),n(`L${C}.mlp.down`,"matvec_residual",C,O,[b1(`L${C}.mlp.down`,O.rows,O.cols,!1,L,0)],0,I*X.hiddenBytes*J+k,k,O.rows,"mlp_down","none");else{let w=x1(Q.mlp);i(`L${C}.mlp.fused`,"mlp_fused",C,null,[b1(`L${C}.mlp.gate_up`,S.rows,S.cols,!1,L,0),b1(`L${C}.mlp.down`,O.rows,O.cols,!1,L,Q.mlp)],M,k+w.foldSlices*L1,k+Q.mlp*L1,w,Q.mlp*J,j.blockWorkgroupSize,null,null,"none")}}j1("head.fold");let W1=d1([H7(L)],y,E),B1=D*X.logitsBytes*J,G1=$.samplePartials.plain;if(n("head","norm_head",-1,W1,[b1("head",W1.rows,W1.cols,!0,L,0)],G*4,k,B1,W1.rows,"head","none"),i("sample.partial","sample_partial",-1,null,[],0,B1,x0*G1,$1(),G1,j.sampleWorkgroupSize,null,null,"none"),i("sample.final","sample_final",-1,null,[],0,x0*G1,x0,$1(),1,j.sampleWorkgroupSize,null,null,"none"),B.residual!==0||B.pending!==0)throw Error(`planStep: token does not close on its starting parity (residual ${B.residual}, pending ${B.pending})`);return R1}function G9(L){let $=X7.find((J)=>J.key===L);if($===void 0)throw Error(`armSpec: unknown arm ${L}`);return $}function q9(L){let $=G9(L);return{splitN:$.splitN,flash:$.flash}}function $0(L){let $=L.ffn/L.quantBlock;if(!Number.isInteger($))throw Error(`planSlicesFor: ffn ${L.ffn} must be a multiple of the quantisation block ${L.quantBlock}`);if(L.hidden%L.heads!==0)throw Error(`planSlicesFor: hidden ${L.hidden} does not divide into ${L.heads} heads`);if(L.heads%L.kvHeads!==0)throw Error(`planSlicesFor: ${L.heads} q heads do not divide into ${L.kvHeads} kv heads`);return{mlp:$}}function _0(L){return $0(L).mlp}var W7=$0(P1),_4=_0(P1);if(f.mlpFused.slices!==W7.mlp)throw Error(`plan_options: mlpFused registers ${f.mlpFused.slices} slices, the shape rule gives ${W7.mlp}`);if(f.flash.coreWorkgroupSize!==P1.headDim)throw Error(`plan_options: the flash core registers workgroup ${f.flash.coreWorkgroupSize} against head dimension ${P1.headDim}`);if(f.flash.mergeWorkgroupSize!==P1.headDim)throw Error(`plan_options: the merge registers workgroup ${f.flash.mergeWorkgroupSize} against head dimension ${P1.headDim}`);if(f.flash.partialWords<P1.headDim+2)throw Error(`plan_options: ${f.flash.partialWords} partial words cannot hold ${P1.headDim} lanes plus a maximum and a denominator`);if(!A0.concatQkv||!A0.concatGateUp)throw Error("plan_options: planStep has no unconcatenated form; FUSION registers both concatenations as true");var A9={plain:g.sample.partials},L0={workgroupSize:g.matvec.workgroupSize,subgroupSize:g.matvec.subgroupSize,rowsPerSubgroup:g.matvec.rowsPerSubgroup,subgroupsPerRow:g.matvec.subgroupsPerRow},k1={workgroupSize:g.matvecWide.workgroupSize,subgroupSize:g.matvecWide.subgroupSize,rowsPerSubgroup:g.matvecWide.rowsPerSubgroup,subgroupsPerRow:g.matvecWide.subgroupsPerRow},O9={conv_in_proj:L0,conv_out_proj:L0,attn_qkv:k1,attn_o_proj:k1,mlp_gate_up:k1,mlp_down:k1,head:L0},F9={attn_o_proj:k1,mlp_down:k1,attn_qkv:k1,conv_in_proj:L0,mlp_gate_up:k1,head:L0};function G7(L){let $=new Map;for(let Z of L){if($.has(Z.shape))throw Error(`knee table: ${Z.shape} appears twice`);$.set(Z.shape,Z.geometry)}let J=(Z)=>{let X=$.get(Z);if(X===void 0)throw Error(`knee table: no measured cell for ${Z}`);return X};return{attn_o_proj:J("attn_o_proj"),mlp_down:J("mlp_down"),attn_qkv:J("attn_qkv"),conv_in_proj:J("conv_in_proj"),mlp_gate_up:J("mlp_gate_up"),head:J("head")}}function C9(L,$){let J=L.flashOptima.find((Z)=>Z.positions===$);if(J!==void 0){if(!Number.isInteger(J.blocks)||J.blocks<1||J.blocks>f.flash.maxBlocks)throw Error(`flashBlocksFor: P0c optimum ${J.blocks} at ${$} positions is outside 1..${f.flash.maxBlocks}`);return J.blocks}return V7($,f.flash.positionsPerBlockRule,f.flash.maxBlocks)}function q7(L){for(let $ of j7){let J=L[$];if(J.workgroupSize%J.subgroupSize!==0)throw Error(`planGeometry: knee for ${$} has workgroup ${J.workgroupSize} against subgroup ${J.subgroupSize}`)}return{carried:O9,knee:L,blockWorkgroupSize:f.mlpFused.workgroupSize,attnCoreWorkgroupSize:f.attnSplit.coreWorkgroupSize,flashCoreWorkgroupSize:f.flash.coreWorkgroupSize,mergeWorkgroupSize:f.flash.mergeWorkgroupSize,convWorkgroupSize:g.conv.workgroupSize,embedWorkgroupSize:g.embed.workgroupSize,sampleWorkgroupSize:g.sample.workgroupSize,foldGeometry:{workgroupSize:f.fold.workgroupSize,threadsPerRow:f.fold.threadsPerRow}}}var m4=q7(F9);function M9(L,$){return{blocks:C9(L,$),maxBlocks:f.flash.maxBlocks,partialWords:f.flash.partialWords}}function A7(L,$,J,Z,X,Y){return{queries:$,position:J,precision:{residualBytes:s.residualBytes,hiddenBytes:s.hiddenBytes,kvCacheBytes:s.kvCacheBytes,convCacheBytes:s.convCacheBytes,logitsBytes:s.logitsBytes},levers:q9(X),slices:$0(L),flash:M9(Y,Z),geometry:q7(Y.knee),samplePartials:A9}}var C0=[];function O7(L){if(!L.length)throw Error("No registered candidates");C0=L}function J0(L){let $=C0.find((J)=>J.key===L);if(!$)throw Error("Unknown candidate "+L);return $}function F7(L,$,J,Z){return C0.map((X)=>{let Y=A7(L,1,$,J,X.unfused?"A2N1":X.flash?"A2N2":"A2",Z),Q={...Y.geometry.carried};for(let[H,j]of Object.entries(X.matvec))Q[H]=j.geometry;return{key:X.key,options:{...Y,flash:{...Y.flash,blocks:X.flashBlocks??Y.flash.blocks},slices:{...Y.slices,mlp:X.mlpSlices??Y.slices.mlp},geometry:{...Y.geometry,foldGeometry:{...Y.geometry.foldGeometry,threadsPerRow:X.foldLanes??Y.geometry.foldGeometry.threadsPerRow},carried:Q,blockWorkgroupSize:X.mlpWorkgroup??Y.geometry.blockWorkgroupSize}}}})}function C7(){return Math.max(80,...C0.map((L)=>L.mlpSlices??80))}var i4={...Z7,port:47320};var{maxComputeWorkgroupStorageSizeFlash:D9,...M7}=Y7,D7={...M7,maxComputeWorkgroupStorageSize:Math.max(M7.maxComputeWorkgroupStorageSize,D9)};function m0(L,$){return $.kind==="mlp_fused"?L.normOnceMlp??!1:["norm_matvec","attn_proj","norm_head"].includes($.kind)?(L.normOnceMatvec||$.kind==="norm_head"&&L.normOnceHead)??!1:!1}function f0(L,$,J,Z,X){let Y=[];for(let Q of L){if($.flashOnline&&Q.kind==="attn_core_flash"){if(X!==1)throw Error("Online flash accounting requires one decode query");let H=$.flashBlocks,j=J.heads*H*J.headDim*4*3,V=J.heads*2*Z*J.headDim*2,G=J.heads*H*(J.headDim+2)*4+J.kvHeads*2*J.headDim*2;Y.push({...Q,index:Y.length,readBytes:j+V,writeBytes:G});continue}if($.convFuse&&Q.kind==="conv_core")continue;if($.convFuse&&Q.kind==="norm_matvec"){Y.push({...Q,index:Y.length,kind:"conv_proj_core",constantBytes:Q.constantBytes+12288,readBytes:Q.readBytes+6144,writeBytes:8192});continue}if(m0($,Q))Y.push({...Q,index:Y.length,name:Q.name+".input_norm",kind:"input_norm",weight:null,reads:[],constantBytes:4096,readBytes:4096,writeBytes:4096,workgroups:1,site:null,geometry:null,fusion:{...Q.fusion,slices:0,foldSlices:0}});Y.push({...Q,index:Y.length})}return Y}var I9=["maxBufferSize","maxStorageBufferBindingSize","maxStorageBuffersPerShaderStage","maxComputeWorkgroupStorageSize","maxComputeInvocationsPerWorkgroup","maxComputeWorkgroupSizeX","maxComputeWorkgroupsPerDimension","maxBindGroups","maxBindingsPerBindGroup"];async function N9(L){if(navigator.gpu===void 0)throw Error("navigator.gpu unavailable");let $=await navigator.gpu.requestAdapter({powerPreference:L});if($===null)throw Error("requestAdapter returned null");return $}function T9(L,$,J){let Z=L.descriptor.adapter;if(Z.isFallbackAdapter)throw Error("SOFTWARE_FALLBACK: adapter.info.isFallbackAdapter");if(Z.subgroupMinSize!==J||Z.subgroupMaxSize!==J)throw Error(`subgroup size ${Z.subgroupMinSize}..${Z.subgroupMaxSize} != compiled ${J}`);for(let X of $.features){let Y=o1(X);if(!L.device.features.has(Y))throw Error(`device lacks required feature ${Y}`)}for(let[X,Y]of Object.entries($.limits)){let Q=Reflect.get(L.device.limits,X);if(typeof Q!=="number")throw Error(`device limit ${X} missing`);if(Q<Y)throw Error(`device limit ${X} = ${Q} < required ${Y}`)}}async function R9(L,$){let J=await N9($),Z=[];for(let V of L.features){let G=o1(V);if(!J.features.has(G))throw Error(`adapter lacks required feature ${G}`);Z.push(G)}let X={};for(let V of I9){let G=J.limits[V];if(typeof G!=="number")throw Error(`adapter limit ${V} missing`);X[V]=G}let Y=globalThis.__probeHooks,Q=Y.devices().length,H=await J.requestDevice({requiredFeatures:Z,requiredLimits:X,label:"probe"}),j=Y.descriptors()[Q];if(j===void 0||Y.devices()[Q]!==H)throw Error("hooks did not record the probe device");return{device:H,descriptor:j,owned:!0}}async function M0(L,$,J){let Z=globalThis.__probeHooks,X=Z.devices();if(X.length>1)throw Error(`SHARED_DEVICE: ${X.length} GPUDevices exist; the session must hold one`);let Y=X[0],Q=Z.descriptors()[0],H=Y===void 0||Q===void 0?await R9(L,$):{device:Y,descriptor:Q,owned:!1};return T9(H,L,J),H}async function c1(L){await L.queue.onSubmittedWorkDone()}async function I7(L,$,J,Z){L.pushErrorScope("validation"),L.pushErrorScope("out-of-memory"),L.pushErrorScope("internal");try{return await Z()}finally{let X=await L.popErrorScope(),Y=await L.popErrorScope(),Q=await L.popErrorScope();for(let H of[Q,Y,X])if(H!==null)$.push(`${J}: ${H.constructor.name}: ${H.message}`)}}function B9(){let L=Number.POSITIVE_INFINITY,$=performance.now();for(let J=0;J<20000;J+=1){let Z=performance.now(),X=Z-$;if(X>0&&X<L)L=X;$=Z}return L}function N7(){return{user_agent:navigator.userAgent,cross_origin_isolated:globalThis.crossOriginIsolated,timer_resolution_ms:B9(),hardware_concurrency:navigator.hardwareConcurrency,webgpu:navigator.gpu!==void 0}}async function Z0(L,$,J,Z){let X=L.createBuffer({size:Z,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,label:"readback"}),Y=L.createCommandEncoder();Y.copyBufferToBuffer($,J,X,0,Z),L.queue.submit([Y.finish()]),await X.mapAsync(GPUMapMode.READ,0,Z);let Q=X.getMappedRange(0,Z).slice(0);return X.unmap(),X.destroy(),Q}async function k0(L,$,J){let Z=J*2*8,X=L.createBuffer({size:Z,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC,label:"timestamps"}),Y=L.createCommandEncoder();Y.resolveQuerySet($,0,J*2,X,0),L.queue.submit([Y.finish()]);let Q=await Z0(L,X,0,Z);X.destroy();let H=new BigUint64Array(Q),j=new Float64Array(J*2);for(let V=0;V<j.length;V+=1){let G=H[V];j[V]=G===void 0?0:Number(G)}return j}function T7(L,$,J){let Z=J==="gate"?2*$:L,X=J==="gate"?L/32:$/32;return{workgroupSize:256,bindings:[{binding:0,name:"weights",access:"read"},{binding:1,name:"destination",access:"read_write"}],code:`
  const ROWS:u32=${Z}u;const BLOCKS:u32=${X}u;
  @group(0) @binding(0) var<storage,read> input_words:array<u32>;
  @group(0) @binding(1) var<storage,read_write> output_words:array<u32>;
  @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3<u32>){
   let i=gid.x;if(i>=ROWS*BLOCKS*4u){return;}${J==="gate"?"let row=i/(BLOCKS*4u);let word=(i/BLOCKS)%4u;let block=i%BLOCKS;let at=(row*BLOCKS+block)*4u+word;":"let block=i/(ROWS*4u);let word=(i/ROWS)%4u;let row=i%ROWS;let at=(block*ROWS+row)*4u+word;"}output_words[i]=input_words[at];
  }`}}function a1(L){return L==="v1_vec4_f16"||L==="v3_both"}function z9(L){return L==="v2_vec4_accumulator"||L==="v3_both"}function z7(L,$){return{workgroupSize:L.workgroupSize,subgroupSize:L.subgroupSize,rowsPerSubgroup:L.rowsPerSubgroup,subgroupsPerRow:L.subgroupsPerRow,maxCols:$}}var O1=`enable f16;
enable subgroups;
`,F1=`
struct State {
  position: u32,
  step: u32,
  queries: u32,
  sample_mode: u32,
  rng: u32,
  freerun: u32,
  advance: u32,
  reserved: u32,
}
`,E9={weights:"array<u32>",scales:"array<f32>",zero_points:"array<u32>",gamma:"array<f32>",residual:"array<f32>",source:"array<f16>",destination:"array<f16>",state:"State",tokens:"array<u32>",cos:"array<f32>",sin:"array<f32>",key_cache:"array<f16>",value_cache:"array<f16>",cache:"array<f16>",logits:"array<f32>",partials:"array<u32>",sampled:"array<u32>",blob:"array<u32>",residual_in:"array<f32>",residual_out:"array<f32>",partials_in:"array<f32>",partials_out:"array<f32>",projection:"array<f32>",flash:"array<f32>"};function C1(L,$){return L.map((J)=>{let Z=$[J.name],X=Z===void 0?E9[J.name]:Z;return`@group(0) @binding(${J.binding}) var<storage, ${J.access}> ${J.name}: ${X};`}).join(`
`)}var g0=`
fn dot8(packed: u32, zp: f32, a0: vec4<f32>, a1: vec4<f32>) -> f32 {
  let lo = vec4<f32>(f32(packed & 15u), f32((packed >> 4u) & 15u), f32((packed >> 8u) & 15u), f32((packed >> 12u) & 15u));
  let hi = vec4<f32>(f32((packed >> 16u) & 15u), f32((packed >> 20u) & 15u), f32((packed >> 24u) & 15u), f32((packed >> 28u) & 15u));
  let z = vec4<f32>(zp, zp, zp, zp);
  return dot(lo - z, a0) + dot(hi - z, a1);
}
`,X0=`
fn dot8h(packed: u32, zp: f16, a0: vec4<f16>, a1: vec4<f16>) -> f16 {
  let lo = vec4<f16>(f16(packed & 15u), f16((packed >> 4u) & 15u), f16((packed >> 8u) & 15u), f16((packed >> 12u) & 15u));
  let hi = vec4<f16>(f16((packed >> 16u) & 15u), f16((packed >> 20u) & 15u), f16((packed >> 24u) & 15u), f16((packed >> 28u) & 15u));
  let z = vec4<f16>(zp, zp, zp, zp);
  return dot(lo - z, a0) + dot(hi - z, a1);
}
`,P9=`
fn hash(seed: u32, index: u32) -> u32 {
  var x = seed ^ (index * 2654435761u);
  x = x ^ (x >> 16u);
  x = x * 2246822519u;
  x = x ^ (x >> 13u);
  x = x * 3266489917u;
  return x ^ (x >> 16u);
}

fn gumbel(seed: u32, index: u32) -> f32 {
  let u = (f32(hash(seed, index) >> 8u) + 0.5) * 5.960464477539063e-8;
  return -log(-log(u));
}
`;function v9(L,$){if(!L)return`fn zeroPoint(row: u32, block: u32) -> f32 { return 8.0; }
`;return`
const ZP_BYTES_PER_ROW: u32 = ${Math.ceil($/2)}u;
fn zeroPoint(row: u32, block: u32) -> f32 {
  let index = row * ZP_BYTES_PER_ROW + block / 2u;
  let byte = (zero_points[index / 4u] >> ((index % 4u) * 8u)) & 255u;
  if ((block & 1u) == 0u) { return f32(byte & 15u); }
  return f32(byte >> 4u);
}
`}function b9(L){return`
fn stageInput(query: u32, local: u32) {
  var sum: f32 = 0.0;
  for (var i = local; i < COLS; i = i + WG) {
    let v = residual[query * COLS + i] * ${L.toFixed(1)};
    stage[i] = v;
    sum = sum + v * v;
  }
  reduce[local] = sum;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(COLS) + EPSILON);
  workgroupBarrier();
  for (var i = local; i < COLS; i = i + WG) {
    stage[i] = stage[i] * inv * gamma[i];
  }
  workgroupBarrier();
}
`}var y9=`
fn stageInput(query: u32, local: u32) {
  for (var i = local; i < COLS; i = i + WG) {
    stage[i] = f32(source[query * COLS + i]);
  }
  workgroupBarrier();
}
`;function S9(L){return L===1?"lane + i * SG":"slice * SG + lane + i * (SG * SUBGROUPS_PER_ROW)"}function E7(L){return L===0?"stage[o]":`stage[o + ${L}u]`}function R7(L){return`vec4<f32>(${[0,1,2,3].map(($)=>E7(L+$)).join(", ")})`}function B7(L){return`vec4<f16>(${[0,1,2,3].map(($)=>`f16(${E7(L+$)})`).join(", ")})`}function w9(L,$){let J=$*8,Z=$*8+4;if(a1(L))return`dot8h(packed.${["x","y","z","w"][$]}, zpv,
        ${B7(J)},
        ${B7(Z)})`;return`dot8(${$===0?"weights[base]":`weights[base + ${$}u]`}, zpv,
        ${R7(J)},
        ${R7(Z)})`}function x9(L,$){let J=$===1?"row: u32, lane: u32":"row: u32, slice: u32, lane: u32",Z=a1(L)?`      let packed = weights[row * BLOCKS + block];
      let zpv = f16(zeroPoint(row, block));
      let o = block * 32u;`:`      let base = row * BLOCKS * 4u + block * 4u;
      let zpv = zeroPoint(row, block);
      let o = block * 32u;`,X=[0,1,2,3].map((V)=>w9(L,V)),Y,Q,H;if(z9(L))Y="  var acc: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);",Q=`      let d = vec4<f32>(
        ${X.map((V)=>a1(L)?`f32(${V})`:V).join(`,
        `)});
      acc = acc + d * scales[row * BLOCKS + block];`,H=`  let total = subgroupAdd(acc);
  return total.x + total.y + total.z + total.w;`;else Y="  var acc: f32 = 0.0;",Q=`      var part: ${a1(L)?"f16 = f16(0.0)":"f32 = 0.0"};
${X.map((V)=>`      part = part + ${V};`).join(`
`)}
      acc = acc + ${a1(L)?"f32(part)":"part"} * scales[row * BLOCKS + block];`,H="  return subgroupAdd(acc);";let j=`
fn rowSum(${J}) -> f32 {
${Y}
  for (var i = 0u; i < BLOCK_ITERS; i = i + 1u) {
    let block = ${S9($)};
    if (block < BLOCKS) {
${Z}
${Q}
    }
  }
${H}
}
`;if($===1)return j;return`${j}
fn rowTotal(row: u32, subgroup: u32, slice: u32, lane: u32) -> f32 {
  let mine = rowSum(row, slice, lane);
  if (lane == 0u) { cross[subgroup] = mine; }
  workgroupBarrier();
  let base = subgroup - slice;
  var total: f32 = 0.0;
  for (var p = 0u; p < SUBGROUPS_PER_ROW; p = p + 1u) {
    total = total + cross[base + p];
  }
  workgroupBarrier();
  return total;
}
`}function _9(L,$,J,Z){let X=L==="norm_swiglu"?J.ffn:$;return Math.ceil(X/O0(Z))}function P7(L,$,J,Z,X){return[_9(L,$,J,Z),X,1]}function v7(L,$,J,Z,X,Y,Q,H){let{workgroupSize:j,subgroupSize:V,rowsPerSubgroup:G,subgroupsPerRow:I}=X;if(j%V!==0)throw Error(`matvecKernel: workgroup ${j} is not a multiple of subgroup ${V}`);if(J%Y.quantBlock!==0)throw Error(`matvecKernel: cols ${J} is not a multiple of block ${Y.quantBlock}`);if(J>X.maxCols)throw Error(`matvecKernel: cols ${J} exceeds the registered stage width ${X.maxCols}`);if(L==="norm_swiglu"&&$!==2*Y.ffn)throw Error(`matvecKernel: swiglu expects ${2*Y.ffn} rows, got ${$}`);if(L==="norm_projection"&&$%Y.headDim!==0)throw Error(`matvecKernel: a q/k/v projection emits whole heads, ${$} rows is not a multiple of ${Y.headDim}`);if(!K7(X,J,Y.quantBlock))throw Error(`matvecKernel: ${I} subgroups of ${V} lanes cannot each take a block of a ${J}-wide row`);let D=O0(X),R=j/V,z=J/Y.quantBlock,T=L==="norm_swiglu"?Y.ffn:$,N=I>1;if(N&&G!==1)throw Error(`matvecKernel: cooperating subgroups own one row each, got ${G} per subgroup`);let y=[{binding:0,name:"weights",access:"read"},{binding:1,name:"scales",access:"read"}];if(Z)y.push({binding:2,name:"zero_points",access:"read"});if(L==="matvec_residual")y.push({binding:4,name:"residual",access:"read_write"}),y.push({binding:5,name:"source",access:"read"});else y.push({binding:3,name:"gamma",access:"read"}),y.push({binding:4,name:"residual",access:"read"});if(L==="norm_head")y.push({binding:16,name:"logits",access:"read_write"});else if(L==="norm_projection")y.push({binding:17,name:"projection",access:"read_write"});else if(L!=="matvec_residual")y.push({binding:6,name:"destination",access:"read_write"});y.push({binding:7,name:"state",access:"read"});let E=a1(H)?{weights:"array<vec4<u32>>"}:{},t=L==="norm_head"?"logits[query * EMITS + row] = value;":L==="norm_projection"?"projection[query * EMITS + row] = value;":L==="matvec_residual"?"residual[query * EMITS + row] = residual[query * EMITS + row] + value;":"destination[query * EMITS + row] = f16(value);",e=N?"rowTotal(ROWEXPR, subgroup, slice, lane)":"rowSum(ROWEXPR, lane)",k=(n)=>e.replace("ROWEXPR",n),h=N?"lane == 0u && slice == 0u":"lane == 0u",L1=L==="norm_swiglu"?`
  for (var r = 0u; r < ROWS_PER_SG; r = r + 1u) {
    let want = rowBase + r;
    let row = min(want, EMITS - 1u);
    let gate = ${k("row")};
    let up = ${k("row + EMITS")};
    if (${h} && want < EMITS) {
      destination[query * EMITS + row] = f16((gate / (1.0 + exp(-gate))) * up);
    }
  }
`:`  for (var r = 0u; r < ROWS_PER_SG; r = r + 1u) {
    let want = rowBase + r;
    let row = min(want, EMITS - 1u);
    let value = ${k("row")};
    if (${h} && want < EMITS) {
      ${t}
    }
  }
`,U1=a1(H)?X0:g0,w1=N?`var<workgroup> cross: array<f32, ${R}>;
`:"",R1=N?"let rowBase = wid.x * ROWS_PER_WG + (subgroup / SUBGROUPS_PER_ROW) * ROWS_PER_SG;":"let rowBase = wid.x * ROWS_PER_WG + (local / SG) * ROWS_PER_SG;",B=N?`  let subgroup = local / SG;
  let slice = subgroup % SUBGROUPS_PER_ROW;
`:"",I1=N?`const SUBGROUPS_PER_ROW: u32 = ${I}u;
`:"";return{code:`${O1}${F1}${U1}
const COLS: u32 = ${J}u;
const BLOCKS: u32 = ${z}u;
const BLOCK_ITERS: u32 = ${Math.ceil(z/(V*I))}u;
const WG: u32 = ${j}u;
const SG: u32 = ${V}u;
${I1}const ROWS_PER_SG: u32 = ${G}u;
const ROWS_PER_WG: u32 = ${D}u;
const EMITS: u32 = ${T}u;
const EPSILON: f32 = ${Y.epsilon};

${C1(y,E)}

var<workgroup> stage: array<f32, ${J}>;
var<workgroup> reduce: array<f32, ${j}>;
${w1}${v9(Z,z)}${L==="matvec_residual"?y9:b9(Q)}${x9(H,I)}
@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) local: u32,
  @builtin(subgroup_invocation_id) lane: u32,
) {
  let query = wid.y;
  if (query >= state.queries) { return; }
  stageInput(query, local);
${B}  ${R1}
${L1}}
`,bindings:y,workgroupSize:j}}function b7(L,$){let J=L.hidden/L.quantBlock,Z=[{binding:0,name:"weights",access:"read"},{binding:1,name:"scales",access:"read"},{binding:2,name:"zero_points",access:"read"},{binding:4,name:"residual",access:"read_write"},{binding:7,name:"state",access:"read"},{binding:8,name:"tokens",access:"read"}];return{code:`${O1}${F1}
const HIDDEN: u32 = ${L.hidden}u;
const BLOCK: u32 = ${L.quantBlock}u;
const BLOCKS: u32 = ${J}u;
const U32_PER_ROW: u32 = ${L.hidden/8}u;
const ZP_BYTES_PER_ROW: u32 = ${Math.ceil(J/2)}u;
const WG: u32 = ${$}u;

${C1(Z,{})}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let query = wid.x;
  if (query >= state.queries) { return; }
  let token = tokens[state.step + query];
  for (var i = local; i < HIDDEN; i = i + WG) {
    let block = i / BLOCK;
    let packed = weights[token * U32_PER_ROW + i / 8u];
    let nibble = (packed >> ((i % 8u) * 4u)) & 15u;
    let zpIndex = token * ZP_BYTES_PER_ROW + block / 2u;
    let zpByte = (zero_points[zpIndex / 4u] >> ((zpIndex % 4u) * 8u)) & 255u;
    var zpv: u32 = zpByte & 15u;
    if ((block & 1u) == 1u) { zpv = zpByte >> 4u; }
    residual[query * HIDDEN + i] = (f32(nibble) - f32(zpv)) * scales[token * BLOCKS + block];
  }
}
`,bindings:Z,workgroupSize:$}}function y7(L){return[L,1,1]}function S7(L,$){if(L.convTaps!==3)throw Error(`convKernel: registered for 3 taps, got ${L.convTaps}`);let J=[{binding:3,name:"gamma",access:"read"},{binding:5,name:"source",access:"read"},{binding:6,name:"destination",access:"read_write"},{binding:7,name:"state",access:"read"},{binding:9,name:"cache",access:"read_write"}];return{code:`${O1}${F1}
const HIDDEN: u32 = ${L.hidden}u;
const TAPS: u32 = ${L.convTaps}u;
const WG: u32 = ${$}u;

${C1(J,{})}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let channel = gid.x;
  if (channel >= HIDDEN) { return; }
  let w0 = gamma[channel * TAPS];
  let w1 = gamma[channel * TAPS + 1u];
  let w2 = gamma[channel * TAPS + 2u];
  var s1 = f32(cache[channel * TAPS + 1u]);
  var s2 = f32(cache[channel * TAPS + 2u]);
  for (var q = 0u; q < state.queries; q = q + 1u) {
    let base = q * 3u * HIDDEN;
    let bx = f32(source[base + channel]) * f32(source[base + 2u * HIDDEN + channel]);
    let y = w0 * s1 + w1 * s2 + w2 * bx;
    destination[q * HIDDEN + channel] = f16(f32(source[base + HIDDEN + channel]) * y);
    s1 = s2;
    s2 = bx;
  }
  cache[channel * TAPS] = f16(0.0);
  cache[channel * TAPS + 1u] = f16(s1);
  cache[channel * TAPS + 2u] = f16(s2);
}
`,bindings:J,workgroupSize:$}}function w7(L,$){return[Math.ceil(L.hidden/$),1,1]}function x7(L,$,J){if(L.vocab%J!==0)throw Error(`samplePartialKernel: vocab ${L.vocab} is not a multiple of ${J}`);let Z=[{binding:7,name:"state",access:"read"},{binding:15,name:"partials",access:"read_write"},{binding:16,name:"logits",access:"read"}];return{code:`${O1}${F1}
const VOCAB: u32 = ${L.vocab}u;
const WG: u32 = ${$}u;
const PER_GROUP: u32 = ${L.vocab/J}u;

${C1(Z,{})}

var<workgroup> bestValue: array<f32, ${$}>;
var<workgroup> bestIndex: array<u32, ${$}>;
${P9}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let offset = (state.queries - 1u) * VOCAB;
  var value: f32 = -3.0e38;
  var index: u32 = 0u;
  for (var i = wid.x * PER_GROUP + local; i < (wid.x + 1u) * PER_GROUP; i = i + WG) {
    var s = logits[offset + i];
    if (state.sample_mode == 1u) { s = s + gumbel(state.rng, i); }
    if (s > value) { value = s; index = i; }
  }
  bestValue[local] = value;
  bestIndex[local] = index;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      if (bestValue[local + stride] > bestValue[local]) {
        bestValue[local] = bestValue[local + stride];
        bestIndex[local] = bestIndex[local + stride];
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    partials[wid.x * 2u] = bitcast<u32>(bestValue[0]);
    partials[wid.x * 2u + 1u] = bestIndex[0];
  }
}
`,bindings:Z,workgroupSize:$}}function _7(L,$,J){let Z=[{binding:7,name:"state",access:"read_write"},{binding:8,name:"tokens",access:"read_write"},{binding:15,name:"partials",access:"read"},{binding:17,name:"sampled",access:"read_write"}];return{code:`${O1}${F1}
const WG: u32 = ${$}u;
const PARTIALS: u32 = ${J}u;
const MAX_POSITIONS: u32 = ${L.maxPositions}u;

${C1(Z,{})}

var<workgroup> bestValue: array<f32, ${$}>;
var<workgroup> bestIndex: array<u32, ${$}>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_index) local: u32) {
  var value: f32 = -3.0e38;
  var index: u32 = 0u;
  for (var i = local; i < PARTIALS; i = i + WG) {
    let candidate = bitcast<f32>(partials[i * 2u]);
    if (candidate > value) { value = candidate; index = partials[i * 2u + 1u]; }
  }
  bestValue[local] = value;
  bestIndex[local] = index;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) {
      if (bestValue[local + stride] > bestValue[local]) {
        bestValue[local] = bestValue[local + stride];
        bestIndex[local] = bestIndex[local + stride];
      }
    }
    workgroupBarrier();
  }
  if (local == 0u) {
    let token = bestIndex[0];
    let last = state.step + state.queries - 1u;
    sampled[last] = token;
    if (state.freerun == 1u && last + 1u < MAX_POSITIONS) {
      tokens[last + 1u] = token;
    }
    if (state.advance == 1u) {
      state.position = state.position + state.queries;
      state.step = state.step + state.queries;
      state.rng = state.rng * 1664525u + 1013904223u;
    }
  }
}
`,bindings:Z,workgroupSize:$}}function m7(L){return[L,1,1]}function f7(){return[1,1,1]}function k7(L,$,J){if(!J)return L;if($>32)throw Error("Subgroup merge supports at most 32 splits");let Z=L.code.indexOf("@compute");if(Z<0)throw Error("Merge source seam");let X=L.code.slice(0,Z)+`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
  let first=(wid.y*SPLITS*HEADS+wid.x)*RECORD;
  var mine=NEG_INF;if(lane<SPLITS){mine=flash[first+lane*HEADS*RECORD+HEAD_DIM];}
  let peak=subgroupMax(mine);var weight=0.0;var den=0.0;
  if(lane<SPLITS){weight=exp(mine-peak);den=weight*flash[first+lane*HEADS*RECORD+HEAD_DIM+1u];}
  let denominator=subgroupAdd(den);var numerator=0.0;
  for(var s=0u;s<SPLITS;s++){numerator+=subgroupShuffle(weight,s)*flash[first+s*HEADS*RECORD+local];}
  destination[(wid.y*HEADS+wid.x)*HEAD_DIM+local]=f16(numerator/denominator);
 }`;return{...L,code:X}}function g7(L,$,J){if(L.headDim!==64||L.heads/L.kvHeads!==2)throw Error("Online flash requires 64-d GQA2");return{workgroupSize:64,bindings:[{binding:0,name:"blob",access:"read"},{binding:5,name:"state",access:"read"},{binding:6,name:"key_cache",access:"read_write"},{binding:7,name:"value_cache",access:"read_write"},{binding:8,name:"projection",access:"read"},{binding:10,name:"flash",access:"read_write"}],code:`enable f16;
 enable subgroups;
 ${F1}
 const SPLITS:u32=${J}u;const MAX_POSITIONS:u32=${L.maxPositions}u;
 @group(0) @binding(0) var<storage,read> blob:array<u32>;
 @group(0) @binding(5) var<storage,read> state:State;
 @group(0) @binding(6) var<storage,read_write> key_cache:array<f16>;
 @group(0) @binding(7) var<storage,read_write> value_cache:array<f16>;
 @group(0) @binding(8) var<storage,read> projection:array<f32>;
 @group(0) @binding(10) var<storage,read_write> flash:array<f32>;
 @compute @workgroup_size(64)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
  let kvHead=wid.x/SPLITS;let split=wid.x%SPLITS;let group=local/32u;let head=kvHead*2u+group;
  let position=subgroupBroadcastFirst(state.position)+wid.y;let total=position+1u;
  let blockLen=(total+SPLITS-1u)/SPLITS;let start=split*blockLen;let stop=min(start+blockLen,total);let owner=(total-1u)/blockLen;
  let base=wid.y*2048u;
  var qlo=projection[base+head*64u+lane];var qhi=projection[base+head*64u+lane+32u];
  let qinv=inverseSqrt(subgroupAdd(qlo*qlo+qhi*qhi)/64.0+${L.epsilon});
  qlo=qlo*qinv*bitcast<f32>(blob[${$.qNorm}u+lane]);qhi=qhi*qinv*bitcast<f32>(blob[${$.qNorm}u+lane+32u]);
  let co=bitcast<f32>(blob[${$.cos}u+position*32u+lane]);let si=bitcast<f32>(blob[${$.sin}u+position*32u+lane]);
  let qrot=qlo*co-qhi*si;qhi=f32(f16(qhi*co+qlo*si));qlo=f32(f16(qrot));
  var klo=projection[base+1024u+kvHead*64u+lane];var khi=projection[base+1024u+kvHead*64u+lane+32u];
  let kinv=inverseSqrt(subgroupAdd(klo*klo+khi*khi)/64.0+${L.epsilon});
  klo=klo*kinv*bitcast<f32>(blob[${$.kNorm}u+lane]);khi=khi*kinv*bitcast<f32>(blob[${$.kNorm}u+lane+32u]);
  let krot=klo*co-khi*si;khi=f32(f16(khi*co+klo*si));klo=f32(f16(krot));
  let vlo=f32(f16(projection[base+1536u+kvHead*64u+lane]));let vhi=f32(f16(projection[base+1536u+kvHead*64u+lane+32u]));
  if(split==owner&&group==0u){let at=(kvHead*MAX_POSITIONS+position)*64u+lane;
   key_cache[at]=f16(klo);key_cache[at+32u]=f16(khi);value_cache[at]=f16(vlo);value_cache[at+32u]=f16(vhi);
  }
  var peak=-3.0e38;var denom=0.0;var nlo=0.0;var nhi=0.0;
  for(var p=start;p<stop;p++){
   var kl=klo;var kh=khi;var vl=vlo;var vh=vhi;
   if(p!=position){let at=(kvHead*MAX_POSITIONS+p)*64u+lane;kl=f32(key_cache[at]);kh=f32(key_cache[at+32u]);vl=f32(value_cache[at]);vh=f32(value_cache[at+32u]);}
   let score=subgroupAdd(qlo*kl+qhi*kh)*0.125;
   let next=max(peak,score);let alpha=exp(peak-next);let beta=exp(score-next);
   denom=denom*alpha+beta;nlo=nlo*alpha+vl*beta;nhi=nhi*alpha+vh*beta;peak=next;
  }
  let rec=((wid.y*SPLITS+split)*16u+head)*68u;
  flash[rec+lane]=nlo;flash[rec+lane+32u]=nhi;
  if(lane==0u){flash[rec+64u]=peak;flash[rec+65u]=denom;}
 }`}}function h0(L,$){if(!$)return L;let J=L.code;for(let Z of["dot8","dot8h"]){if(!J.includes("fn "+Z+"("))continue;let X=Z==="dot8h",Y=X?"f16":"f32",Q=Array.from({length:4},(H,j)=>{let V=`(((packed >> ${j*4}u)&0x000f000fu)|0x64006400u)`,G=X?`bitcast<vec2<f16>>(${V})`:`unpack2x16float(${V})`;return`let p${j}=${G}-vec2<${Y}>(${Y}(1024.0)+zp);`}).join(`
`);J=n1(J,Z,`fn ${Z}(packed:u32,zp:${Y},a0:vec4<${Y}>,a1:vec4<${Y}>)->${Y}{${Q}return dot(vec4<${Y}>(p0.x,p1.x,p2.x,p3.x),a0)+dot(vec4<${Y}>(p0.y,p1.y,p2.y,p3.y),a1);}`)}return{...L,code:J}}function u0(L,$){if(!$||!L.code.includes("fn dot8("))return L;let J=n1(L.code,"dot8",`fn dot8(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
  let mask=vec4<u32>(15u,240u,3840u,61440u);
  let factor=vec4<f32>(1.0,16.0,256.0,4096.0);let inv=vec4<f32>(1.0,0.0625,0.00390625,0.000244140625);
  let lo=vec4<f32>(vec4<u32>(packed)&mask)-vec4<f32>(zp)*factor;
  let hi=vec4<f32>(vec4<u32>(packed>>16u)&mask)-vec4<f32>(zp)*factor;
  return dot(lo,a0*inv)+dot(hi,a1*inv);
 }`);return{...L,code:J}}function n1(L,$,J){let Z=L.indexOf("fn "+$+"(");if(Z<0)throw Error("Missing function "+$);let X=L.indexOf("{",Z)+1,Y=1;while(Y&&X<L.length){if(L[X]==="{")Y++;if(L[X]==="}")Y--;X++}if(Y)throw Error("Unclosed function "+$);return L.slice(0,Z)+J+L.slice(X)}function h7(L){let $=0;if(L=L.replace(/\bstage\[o(?:\s*\+\s*(\d+)u)?\]/g,(J,Z)=>{return $++,`laneBlock[${Z??"0"}u]`}),$!==32)throw Error("Private input source seam "+$);return L+`
var<private> laneBlock:array<f32,32>;
`}function u7(L,$,J,Z){if(!Z)return L;if(!L.code.includes("const COLS: u32 = 1024u;"))throw Error("Private matvec requires 1024 columns");let X=$==="matvec_residual"?"":`var square=0.0;for(var i=lane;i<COLS;i+=SG){let v=residual[query*COLS+i]*${J.toFixed(1)};square+=v*v;}let inv=inverseSqrt(subgroupAdd(square)/f32(COLS)+EPSILON);`,Y=$==="matvec_residual"?"f32(source[query*COLS+at])":`(residual[query*COLS+at]*${J.toFixed(1)})*inv*gamma[at]`,Q=n1(L.code,"stageInput",`fn stageInput(query:u32,local:u32){let lane=local%SG;${X}for(var i=0u;i<32u;i++){let at=lane*32u+i;laneBlock[i]=${Y};}}`);return{...L,code:h7(Q)}}function l7(L,$){if(!$)return L;return{...L,code:n1(L.code,"stageInput","fn stageInput(query:u32,local:u32){for(var i=local;i<COLS;i+=WG){stage[i]=residual[query*COLS+i];}workgroupBarrier();}")}}function p7(L,$){if(!$)return L;let J=L.code,Z=`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32,@builtin(subgroup_invocation_id) lane:u32){
 let query=wid.y;if(query>=state.queries){return;}stageInput(query,local);
 let rowBase=wid.x*(ROWS_PER_WG/3u)+(local/SG)*(ROWS_PER_SG/3u);
 for(var r=0u;r<ROWS_PER_SG/3u;r++){
  let want=rowBase+r;let channel=min(want,1023u);
  let b=f32(f16(rowSum(channel,lane)));let c=f32(f16(rowSum(channel+1024u,lane)));let x=f32(f16(rowSum(channel+2048u,lane)));
  if(lane==0u&&want<1024u){
   let s1=f32(cache[channel*3u+1u]);let s2=f32(cache[channel*3u+2u]);let bx=b*x;
   let w0=bitcast<f32>(blob[channel*3u]);let w1=bitcast<f32>(blob[channel*3u+1u]);let w2=bitcast<f32>(blob[channel*3u+2u]);
   let y=w0*s1+w1*s2+w2*bx;destination[query*1024u+channel]=f16(c*y);
   cache[channel*3u]=f16(0.0);cache[channel*3u+1u]=f16(s2);cache[channel*3u+2u]=f16(bx);
  }
 }
 }`,X=J.indexOf("@compute");if(X<0)throw Error("Conv projection source seam");return J=J.slice(0,X)+Z+`
@group(0) @binding(18) var<storage,read> blob:array<u32>;
@group(0) @binding(9) var<storage,read_write> cache:array<f16>;
`,{...L,code:J,bindings:[...L.bindings,{binding:18,name:"blob",access:"read"},{binding:9,name:"cache",access:"read_write"}]}}function d7(L,$){return{workgroupSize:L,bindings:[{binding:0,name:"residual",access:"read"},{binding:1,name:"gamma",access:"read"},{binding:2,name:"projection",access:"read_write"}],code:`enable subgroups;
 const WG:u32=${L}u;
 @group(0) @binding(0) var<storage,read> residual:array<f32>;
 @group(0) @binding(1) var<storage,read> gamma:array<f32>;
 @group(0) @binding(2) var<storage,read_write> projection:array<f32>;
 var<workgroup> squares:array<f32,${L/32}>;
 @compute @workgroup_size(WG) fn main(@builtin(local_invocation_index) local:u32){
  var sum=0.0;for(var i=local;i<1024u;i+=WG){let v=residual[i]*${$.toFixed(1)};sum+=v*v;}
  let sg=subgroupAdd(sum);if(local%32u==0u){squares[local/32u]=sg;}workgroupBarrier();
  var total=0.0;for(var g=0u;g<WG/32u;g++){total+=squares[g];}
  let inv=inverseSqrt(total/1024.0+0.00001);
  for(var i=local;i<1024u;i+=WG){projection[i]=(residual[i]*${$.toFixed(1)})*inv*gamma[i];}
 }`}}function l0(L,$){if(!$)return L;let J=0,Z=L.code.replace(/\bstage\[([^\]]+)\]/g,(X,Y)=>{return J++,`stage[stageIndex(${Y})]`});if(J<8)throw Error("Stage swizzle source seam");return{...L,code:Z+`
fn stageIndex(i:u32)->u32{return i ^ ((i >> 5u) & 31u);}
`}}function c7(L,$){let J=L.code;if($.mlpVecLoad){if($.mlpPair||$.projectPair)throw Error("Vector-load arm uses unpaired baseline arithmetic");let Z=0;if(J=J.replace(/var part: f32 = 0\.0;/g,(X)=>`let packed = weights[base / 4u];
      `+X),J=J.replace(/blob\[base(?: \+ ([123])u)?\]/g,(X,Y)=>{return Z++,"packed."+["x","y","z","w"][Number(Y??0)]}),Z!==8)throw Error("MLP vector load seam "+Z);J+=`
@group(0) @binding(19) var<storage,read> weights:array<vec4<u32>>;
`,L={...L,bindings:[...L.bindings,{binding:19,name:"weights",access:"read"}]}}if($.normOnceMlp)J=n1(J,"stageBlock",`fn stageBlock(local:u32,group:u32){
   let rowBegin=group*HIDDEN/SLICES;let rowEnd=(group+1u)*HIDDEN/SLICES;
   for(var i=rowBegin+local;i<rowEnd;i+=WG){residual_out[i]=residual_in[i];}
   for(var i=local;i<HIDDEN;i+=WG){stage[i]=projection[i];}workgroupBarrier();
  }`),J+=`
@group(0) @binding(8) var<storage,read> projection:array<f32>;
`,L={...L,bindings:[...L.bindings,{binding:8,name:"projection",access:"read"}]};if($.mlpPrivate){if($.mlpPair||!J.includes("const FOLD_SLICES: u32 = 0u;"))throw Error("Private MLP requires a plain no-fold input");J=n1(J,"stageBlock",`fn stageBlock(local:u32,group:u32){
   let lane=local%SG;var square=0.0;
   for(var i=lane;i<HIDDEN;i+=SG){let v=residual_in[i];square+=v*v;}
   let inv=inverseSqrt(subgroupAdd(square)/f32(HIDDEN)+EPSILON);
   let rowBegin=group*HIDDEN/SLICES;let rowEnd=(group+1u)*HIDDEN/SLICES;
   for(var i=rowBegin+local;i<rowEnd;i+=WG){residual_out[i]=residual_in[i];}
   for(var i=0u;i<32u;i++){let at=lane*32u+i;laneBlock[i]=residual_in[at]*inv*bitcast<f32>(blob[NORM+at]);}
  }`),J=h7(J)}if($.mlpPair){let Z=J.indexOf("fn rowSum("),X=J.indexOf("const KQUANT:",Z);if(Z<0||X<0)throw Error("PAIR source seam");let Y=Array.from({length:4},(Q,H)=>{let j=H*8,V=Array.from({length:4},(I,D)=>`stage[o+${j+D}u]`).join(","),G=Array.from({length:4},(I,D)=>`stage[o+${j+D+4}u]`).join(",");return`let x${H}=vec4<f32>(${V}); let y${H}=vec4<f32>(${G});
part.x += dot8(blob[base+${H}u],8.0,x${H},y${H});
part.y += dot8(blob[upbase+${H}u],8.0,x${H},y${H});`}).join(`
`);J=J.slice(0,Z)+`fn rowPair(row:u32,lane:u32)->vec2<f32>{
 var acc=vec2<f32>(0.0);
 for(var i=0u;i<BLOCK_ITERS;i++){
 let block=lane+i*SG;
 if(block<BLOCKS){let base=QUANT+row*BLOCKS*4u+block*4u;let upbase=base+FFN*BLOCKS*4u;let o=block*32u;var part=vec2<f32>(0.0);
 ${Y}
 acc += part*vec2<f32>(bitcast<f32>(blob[SCALES+row*BLOCKS+block]),bitcast<f32>(blob[SCALES+(FFN+row)*BLOCKS+block]));}
 }
 return subgroupAdd(acc);
}
`+J.slice(X),J=J.replace(`let gate = rowSum(rowBase + r, lane);
    let up = rowSum(FFN + rowBase + r, lane);`,`let pair = rowPair(rowBase + r, lane);
    let gate = pair.x; let up = pair.y;`)}if($.projectPair){let Z=J.indexOf("fn project("),X=J.indexOf("@compute",Z);if(Z<0||X<0)throw Error("PROJECT source seam");let Y=Array.from({length:4},(Q,H)=>{let j=Array.from({length:4},(G,I)=>`hidden_slice[o+${H*8+I}u]`).join(","),V=Array.from({length:4},(G,I)=>`hidden_slice[o+${H*8+I+4}u]`).join(",");return`let x${H}=vec4<f32>(${j});let y${H}=vec4<f32>(${V});part.x+=dot8(blob[base+${H}u],8.0,x${H},y${H});part.y+=dot8(blob[base2+${H}u],8.0,x${H},y${H});`}).join(`
`);J=J.slice(0,Z)+`fn project(local:u32,group:u32){
 let sliceBase=group*HIDDEN;
 for(var row=local;row<HIDDEN;row+=2u*WG){var acc=vec2<f32>(0.0);
 for(var b=0u;b<BLOCKS_PER_SLICE;b++){let base=KQUANT+(sliceBase+row)*WORDS_PER_ROW_SLICE+b*4u;let base2=base+WG*WORDS_PER_ROW_SLICE;let o=b*32u;var part=vec2<f32>(0.0);${Y}
 acc+=part*vec2<f32>(bitcast<f32>(blob[KSCALES+(sliceBase+row)*BLOCKS_PER_SLICE+b]),bitcast<f32>(blob[KSCALES+(sliceBase+row+WG)*BLOCKS_PER_SLICE+b]));}
 partials_out[sliceBase+row]=acc.x;partials_out[sliceBase+row+WG]=acc.y;}
}
`+J.slice(X)}if($.mlpDotIlp){if(!$.mlpPair||$.projectPair||$.mlpVecLoad)throw Error("MLP ILP requires paired gate and plain down");if($.mlpDotIlp==="gate"||$.mlpDotIlp==="both"){let Z=0;if(J=J.replace("var part=vec2<f32>(0.0);","var gatePart=vec4<f32>(0.0);var upPart=vec4<f32>(0.0);"),J=J.replace(/part\.([xy]) \+= dot8\(blob\[(base|upbase)\+(\d)u\],8\.0,x(\d),y(\d)\);/g,(X,Y,Q,H,j,V)=>{if(H!==j||H!==V)throw Error("Gate ILP lane mismatch");return Z++,`${Y==="x"?"gatePart":"upPart"}.${["x","y","z","w"][Number(H)]}=dot8(blob[${Y==="x"?"base":"upbase"}+${H}u],8.0,x${H},y${H});`}),Z!==8)throw Error("Gate ILP source seam "+Z);J=J.replace("acc += part*vec2<f32>","let part=vec2<f32>((gatePart.x+gatePart.y)+(gatePart.z+gatePart.w),(upPart.x+upPart.y)+(upPart.z+upPart.w));acc += part*vec2<f32>")}if($.mlpDotIlp==="down"||$.mlpDotIlp==="both"){let Z=Array.from({length:4},(X,Y)=>{let Q=Array.from({length:4},(j,V)=>`hidden_slice[o+${Y*8+V}u]`).join(","),H=Array.from({length:4},(j,V)=>`hidden_slice[o+${Y*8+V+4}u]`).join(",");return`dot8(blob[base+${Y}u],8.0,vec4<f32>(${Q}),vec4<f32>(${H}))`}).join(",");J=n1(J,"project",`fn project(local:u32,group:u32){
    let sliceBase=group*HIDDEN;
    for(var row=local;row<HIDDEN;row+=WG){var acc=0.0;
     for(var b=0u;b<BLOCKS_PER_SLICE;b++){let base=KQUANT+(sliceBase+row)*WORDS_PER_ROW_SLICE+b*4u;let o=b*32u;
      let parts=vec4<f32>(${Z});let part=(parts.x+parts.y)+(parts.z+parts.w);
      acc+=part*bitcast<f32>(blob[KSCALES+(sliceBase+row)*BLOCKS_PER_SLICE+b]);
     }partials_out[sliceBase+row]=acc;
    }
   }`)}}if($.mlpHalfProduct){J+=`
fn dot8halfProduct(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
   let lo=vec4<f16>(f16(packed&15u),f16((packed>>4u)&15u),f16((packed>>8u)&15u),f16((packed>>12u)&15u))-vec4<f16>(f16(zp));
   let hi=vec4<f16>(f16((packed>>16u)&15u),f16((packed>>20u)&15u),f16((packed>>24u)&15u),f16((packed>>28u)&15u))-vec4<f16>(f16(zp));
   let p0=vec4<f32>(lo*vec4<f16>(a0));let p1=vec4<f32>(hi*vec4<f16>(a1));
   return dot(p0,vec4<f32>(1.0))+dot(p1,vec4<f32>(1.0));
  }`;let Z=J.indexOf($.mlpPair?"fn rowPair(":"fn rowSum("),X=J.indexOf("fn project("),Y=J.indexOf("@compute",X);if(Z<0||X<0||Y<0)throw Error("Half product source seam");if($.mlpHalfProduct==="down"||$.mlpHalfProduct==="both")J=J.slice(0,X)+J.slice(X,Y).replaceAll("dot8(","dot8halfProduct(")+J.slice(Y);if($.mlpHalfProduct==="gate"||$.mlpHalfProduct==="both")J=J.slice(0,Z)+J.slice(Z,X).replaceAll("dot8(","dot8halfProduct(")+J.slice(X)}if($.mlpHalfDot){J+=X0+`
fn dot8half(packed:u32,zp:f32,a0:vec4<f32>,a1:vec4<f32>)->f32{
   return f32(dot8h(packed,f16(zp),vec4<f16>(a0),vec4<f16>(a1)));
  }`;let Z=J.indexOf($.mlpPair?"fn rowPair(":"fn rowSum("),X=J.indexOf("fn project("),Y=J.indexOf("@compute",X);if(Z<0||X<0||Y<0)throw Error("Half dot source seam");if($.mlpHalfDot==="down"||$.mlpHalfDot==="both")J=J.slice(0,X)+J.slice(X,Y).replaceAll("dot8(","dot8half(")+J.slice(Y);if($.mlpHalfDot==="gate"||$.mlpHalfDot==="both")J=J.slice(0,Z)+J.slice(Z,X).replaceAll("dot8(","dot8half(")+J.slice(X)}if($.mlpSoA){if($.mlpVecLoad||$.mlpSlices===160)throw Error("SoA requires ordinary whole quant blocks and no competing weight binding");let Z=J.indexOf($.mlpPair?"fn rowPair(":"fn rowSum("),X=J.indexOf("const KQUANT:",Z),Y=J.indexOf("fn project("),Q=J.indexOf("@compute",Y);if(Z<0||X<0||Y<0||Q<0)throw Error("SoA source seam");if($.mlpSoA==="down"||$.mlpSoA==="both"){let H=0,j=J.slice(Y,Q).replace(/blob\[(base|base2)(?:\s*\+\s*(\d)u)?\]/g,(V,G,I)=>{return H++,`down_soa[((group*BLOCKS_PER_SLICE+b)*4u+${I??0}u)*HIDDEN+row${G==="base2"?"+WG":""}]`});if(H!==($.projectPair?8:4))throw Error("SoA down reads "+H);J=J.slice(0,Y)+j+J.slice(Q),J+=`
@group(0) @binding(20) var<storage,read> down_soa:array<u32>;
`,L={...L,bindings:[...L.bindings,{binding:20,name:"source",access:"read"}]}}if($.mlpSoA==="gate"||$.mlpSoA==="both"){let H=0,j=J.slice(Z,X).replace(/blob\[(base|upbase)(?:\s*\+\s*(\d)u)?\]/g,(V,G,I)=>{return H++,`gate_soa[(${G==="upbase"?"row+FFN":"row"})*BLOCKS*4u+${I??0}u*BLOCKS+block]`});if(H!==($.mlpPair?8:4))throw Error("SoA gate reads "+H);J=J.slice(0,Z)+j+J.slice(X),J+=`
@group(0) @binding(19) var<storage,read> gate_soa:array<u32>;
`,L={...L,bindings:[...L.bindings,{binding:19,name:"weights",access:"read"}]}}}if($.mlpTranspose){if($.mlpStageSwizzle||$.mlpPrivate)throw Error("Transposed stage cannot combine with alternate stage addressing");let Z=0;if(J=J.replace(/\bstage\[o(?:\s*\+\s*(\d+)u)?\]/g,(X,Y)=>{return Z++,`stage[block+${Number(Y??0)*32}u]`}),Z!==32)throw Error("Transposed MLP input seam "+Z);J=J.replace(/\bstage\[i\]/g,"stage[((i&31u)<<5u)+(i>>5u)]")}return h0(u0(l0({...L,code:J},$.mlpStageSwizzle??!1),$.maskMlp??!1),$.magicMlp??!1)}function a7(L,$){if($.foldSubgroup){let X=L.code.indexOf("var<workgroup> part:");if(X<0)throw Error("Fold subgroup seam");let Y=L.code.slice(0,X)+`var<workgroup> part:array<f32,${L.workgroupSize}>;
   @compute @workgroup_size(WG)
   fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32){
    let slot=local%ROWS_PER_WG;let lane=local/ROWS_PER_WG;let row=wid.x*ROWS_PER_WG+slot;
    var acc=0.0;if(row<HIDDEN){for(var s=lane;s<FOLD_SLICES;s+=LANES){acc+=partials_in[s*HIDDEN+row];}}
    for(var delta=ROWS_PER_WG;delta<32u;delta*=2u){acc+=subgroupShuffleXor(acc,delta);}
    if(local%32u<ROWS_PER_WG){part[(local/32u)*ROWS_PER_WG+slot]=acc;}workgroupBarrier();
    if(local<ROWS_PER_WG&&row<HIDDEN){var sum=residual_out[row];for(var sg=0u;sg<WG/32u;sg++){sum+=part[sg*ROWS_PER_WG+local];}residual_out[row]=sum;}
   }`;return{...L,code:Y}}if(!$.foldFast)return L;let J=L.code.indexOf("var<workgroup> part:");if(J<0)throw Error("FOLD source seam");let Z=L.code.slice(0,J)+`@compute @workgroup_size(WG)
 fn main(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) local:u32){
 if(local>=ROWS_PER_WG){return;}let row=wid.x*ROWS_PER_WG+local;if(row>=HIDDEN){return;}
 var sum=residual_out[row];for(var lane=0u;lane<LANES;lane++){var acc=0.0;for(var s=lane;s<FOLD_SLICES;s+=LANES){acc+=partials_in[s*HIDDEN+row];}sum+=acc;}residual_out[row]=sum;
 }`;return{...L,code:Z}}var c={blob:0,residualIn:1,residualOut:2,partialsIn:3,partialsOut:4,state:5,keyCache:6,valueCache:7,projection:8,destination:9,flash:10};function m9(L,$,J,Z){let{workgroupSize:X,subgroupSize:Y,slices:Q,foldSlices:H}=J;if(X%Y!==0)throw Error(`${L}: workgroup ${X} is not a multiple of subgroup ${Y}`);if(Z%Q!==0)throw Error(`${L}: ${Z} is not divisible by ${Q} slices`);let j=Z/Q,V=X/Y;if(j%V!==0)throw Error(`${L}: ${j} rows per slice is not a multiple of the ${V} rows a workgroup sums at once`);if($.hidden%$.quantBlock!==0)throw Error(`${L}: hidden ${$.hidden} is not a multiple of block ${$.quantBlock}`);if(!Number.isInteger(H)||H<0)throw Error(`${L}: fold slices must be a non-negative integer, got ${H}`)}function n7(L,$){let J=(Y)=>Y===0?`${L}[o]`:`${L}[o + ${Y}u]`,Z=(Y)=>`vec4<f32>(${J(Y)}, ${J(Y+1)}, ${J(Y+2)}, ${J(Y+3)})`,X=[];for(let Y=0;Y<4;Y+=1){let Q=Y===0?"base":`base + ${Y}u`;X.push(`${$}part = part + dot8(blob[${Q}], 8.0,
${$}  ${Z(Y*8)},
${$}  ${Z(Y*8+4)});`)}return X.join(`
`)}function f9(L,$,J,Z){return`
const QUANT: u32 = ${L}u;
const SCALES: u32 = ${$}u;
const BLOCKS: u32 = ${J}u;
const BLOCK_ITERS: u32 = ${Math.ceil(J/Z)}u;

fn rowSum(row: u32, lane: u32) -> f32 {
  var acc: f32 = 0.0;
  for (var i = 0u; i < BLOCK_ITERS; i = i + 1u) {
    let block = lane + i * SG;
    if (block < BLOCKS) {
      let base = QUANT + row * BLOCKS * 4u + block * 4u;
      let o = block * 32u;
      var part: f32 = 0.0;
${n7("stage","      ")}
      acc = acc + part * bitcast<f32>(blob[SCALES + row * BLOCKS + block]);
    }
  }
  return subgroupAdd(acc);
}
`}function k9(L,$,J,Z){return`
const KQUANT: u32 = ${L}u;
const KSCALES: u32 = ${$}u;
const BLOCKS_PER_SLICE: u32 = ${J}u;
const WORDS_PER_ROW_SLICE: u32 = ${J*4}u;

fn project(local: u32, group: u32) {
  let sliceBase = group * HIDDEN;
  for (var row = local; row < HIDDEN; row = row + WG) {
    var acc: f32 = 0.0;
    for (var b = 0u; b < BLOCKS_PER_SLICE; b = b + 1u) {
      let base = KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u;
      let o = b * 32u;
      var part: f32 = 0.0;
${n7(Z,"      ")}
      acc = acc + part * bitcast<f32>(blob[KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b]);
    }
    partials_out[sliceBase + row] = acc;
  }
}
`}function g9(L,$,J){let Z=J===0?"    let v = residual_in[i];":`    var v = residual_in[i];
    for (var s = 0u; s < FOLD_SLICES; s = s + 1u) { v = v + partials_in[s * HIDDEN + i]; }`;return`
const NORM: u32 = ${$}u;
const FOLD_SLICES: u32 = ${J}u;
const EPSILON: f32 = ${L.epsilon};

fn stageBlock(local: u32, group: u32) {
  var sum: f32 = 0.0;
  for (var i = local; i < HIDDEN; i = i + WG) {
${Z}
    stage[i] = v;
    sum = sum + v * v;
  }
  reduce[local] = sum;
  workgroupBarrier();
  let rowBegin = (group * HIDDEN) / SLICES;
  let rowEnd = ((group + 1u) * HIDDEN) / SLICES;
  for (var i = rowBegin + local; i < rowEnd; i = i + WG) {
    residual_out[i] = stage[i];
  }
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(HIDDEN) + EPSILON);
  workgroupBarrier();
  for (var i = local; i < HIDDEN; i = i + WG) {
    stage[i] = stage[i] * inv * bitcast<f32>(blob[NORM + i]);
  }
  workgroupBarrier();
}
`}function h9(L,$){return`const HIDDEN: u32 = ${L.hidden}u;
const WG: u32 = ${$.workgroupSize}u;
const SG: u32 = ${$.subgroupSize}u;
const SLICES: u32 = ${$.slices}u;
const ROWS_AT_ONCE: u32 = ${$.workgroupSize/$.subgroupSize}u;
`}function u9(L){let $=[{binding:c.blob,name:"blob",access:"read"},{binding:c.residualIn,name:"residual_in",access:"read"},{binding:c.residualOut,name:"residual_out",access:"read_write"}];if(L.foldSlices>0)$.push({binding:c.partialsIn,name:"partials_in",access:"read"});return $.push({binding:c.partialsOut,name:"partials_out",access:"read_write"}),$}function r7(L,$,J){let Z=L.heads/L.kvHeads;return`const HEAD_DIM: u32 = ${L.headDim}u;
const HALF_DIM: u32 = ${L.headDim/2}u;
const GROUP: u32 = ${Z}u;
const GROUP_ROWS: u32 = ${Z+1}u;
const K_ROW: u32 = ${Z*L.headDim}u;
const KV_WIDTH: u32 = ${L.kvHeads*L.headDim}u;
const MAX_POSITIONS: u32 = ${L.maxPositions}u;
const POS_GROUPS: u32 = ${J/L.headDim}u;
const SCALE: f32 = ${1/Math.sqrt(L.headDim)};
const Q_NORM: u32 = ${$.qNorm}u;
const K_NORM: u32 = ${$.kNorm}u;
const COS: u32 = ${$.cos}u;
const SIN: u32 = ${$.sin}u;
`}function i7(L,$,J){if($.heads%$.kvHeads!==0)throw Error(`${L}: ${$.heads} q heads is not a multiple of ${$.kvHeads} kv heads`);if(J%$.headDim!==0)throw Error(`${L}: workgroup ${J} is not a multiple of head dim ${$.headDim}`);if($.headDim%2!==0)throw Error(`${L}: head dim ${$.headDim} does not split into the two halves the rotation pairs`);return $.heads/$.kvHeads}function t7(){return`
fn headRows(local: u32, position: u32) {
  for (var which = 0u; which < GROUP_ROWS; which = which + 1u) {
    let normBase = select(K_NORM, Q_NORM, which < GROUP);
    let off = which * HEAD_DIM;
    reduce[local] = 0.0;
    workgroupBarrier();
    if (local < HEAD_DIM) { reduce[local] = qk[off + local] * qk[off + local]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    let inv = inverseSqrt(reduce[0] / f32(HEAD_DIM) + EPSILON);
    workgroupBarrier();
    if (local < HEAD_DIM) { qk[off + local] = qk[off + local] * inv * bitcast<f32>(blob[normBase + local]); }
    workgroupBarrier();
    if (local < HALF_DIM) {
      let c = bitcast<f32>(blob[COS + position * HALF_DIM + local]);
      let s = bitcast<f32>(blob[SIN + position * HALF_DIM + local]);
      let lo = qk[off + local];
      let hi = qk[off + local + HALF_DIM];
      reduce[local] = lo * c - hi * s;
      reduce[local + HALF_DIM] = hi * c + lo * s;
    }
    workgroupBarrier();
    if (local < HEAD_DIM) { qk[off + local] = f32(f16(reduce[local])); }
    workgroupBarrier();
  }
}
`}function l9(L,$){return`${t7()}
fn attnCore(local: u32, kvHead: u32, position: u32, queryIndex: u32) {
  let total = position + 1u;
  headRows(local, position);

  // One workgroup per KV head, so the current slot is written exactly once and read by nobody: the
  // walks below score and weight the current position against the k and v still in registers.
  if (local < HEAD_DIM) {
    key_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(qk[K_ROW + local]);
    value_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(vcur[local]);
  }

  var best: array<f32, ${L}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { best[g] = -3.0e38; }
  for (var p = local; p < total; p = p + WG) {
    var acc: array<f32, ${L}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { acc[g] = 0.0; }
    let base = (kvHead * MAX_POSITIONS + p) * HEAD_DIM;
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
      var kd: f32 = qk[K_ROW + d];
      if (p != position) { kd = f32(key_cache[base + d]); }
      for (var g = 0u; g < GROUP; g = g + 1u) { acc[g] = acc[g] + qk[g * HEAD_DIM + d] * kd; }
    }
    for (var g = 0u; g < GROUP; g = g + 1u) {
      let s = acc[g] * SCALE;
      scores[g * MAX_POSITIONS + p] = s;
      best[g] = max(best[g], s);
    }
  }

  var denom: array<f32, ${L}>;
  for (var g = 0u; g < GROUP; g = g + 1u) {
    reduce[local] = best[g];
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
      workgroupBarrier();
    }
    let maximum = reduce[0];
    workgroupBarrier();
    var sum: f32 = 0.0;
    for (var p = local; p < total; p = p + WG) {
      let e = exp(scores[g * MAX_POSITIONS + p] - maximum);
      scores[g * MAX_POSITIONS + p] = e;
      sum = sum + e;
    }
    reduce[local] = sum;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    denom[g] = reduce[0];
    workgroupBarrier();
  }

  // One pass over the value cache for all GROUP heads, weights held in registers, then one fold per
  // head through the shared array in position-group order so the sum is fixed.
  let dim = local % HEAD_DIM;
  let posGroup = local / HEAD_DIM;
  var weighted: array<f32, ${L}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { weighted[g] = 0.0; }
  for (var p = posGroup; p < total; p = p + POS_GROUPS) {
    var vd: f32 = vcur[dim];
    if (p != position) { vd = f32(value_cache[(kvHead * MAX_POSITIONS + p) * HEAD_DIM + dim]); }
    for (var g = 0u; g < GROUP; g = g + 1u) { weighted[g] = weighted[g] + scores[g * MAX_POSITIONS + p] * vd; }
  }
  for (var g = 0u; g < GROUP; g = g + 1u) {
    workgroupBarrier();
    vacc[posGroup * HEAD_DIM + dim] = weighted[g];
    workgroupBarrier();
    if (local < HEAD_DIM) {
      var acc: f32 = 0.0;
      for (var pg = 0u; pg < POS_GROUPS; pg = pg + 1u) { acc = acc + vacc[pg * HEAD_DIM + local]; }
      ${$}
    }
  }
  workgroupBarrier();
}
`}function s7(L,$,J){let Z=i7("attnCoreQkvKernel",L,J),X=[{binding:c.blob,name:"blob",access:"read"},{binding:c.state,name:"state",access:"read"},{binding:c.keyCache,name:"key_cache",access:"read_write"},{binding:c.valueCache,name:"value_cache",access:"read_write"},{binding:c.projection,name:"projection",access:"read"},{binding:c.destination,name:"destination",access:"read_write"}],Y="destination[(queryIndex * HEADS + kvHead * GROUP + g) * HEAD_DIM + local] = f16(acc / denom[g]);";return{code:`${O1}${F1}
const HIDDEN: u32 = ${L.hidden}u;
const HEADS: u32 = ${L.heads}u;
const WG: u32 = ${J}u;
const EPSILON: f32 = ${L.epsilon};
const PROJ_ROWS: u32 = ${L.hidden+2*L.kvHeads*L.headDim}u;
${r7(L,$,J)}
${C1(X,{})}

var<workgroup> reduce: array<f32, ${J}>;
var<workgroup> scores: array<f32, ${Z*L.maxPositions}>;
var<workgroup> qk: array<f32, ${(Z+1)*L.headDim}>;
var<workgroup> vcur: array<f32, ${L.headDim}>;
var<workgroup> vacc: array<f32, ${J}>;
${l9(Z,"destination[(queryIndex * HEADS + kvHead * GROUP + g) * HEAD_DIM + local] = f16(acc / denom[g]);")}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let kvHead = wid.x;
  let queryIndex = wid.y;
  let position = state.position + queryIndex;
  let base = queryIndex * PROJ_ROWS;
  if (local < HEAD_DIM) {
    for (var g = 0u; g < GROUP; g = g + 1u) {
      qk[g * HEAD_DIM + local] = projection[base + (kvHead * GROUP + g) * HEAD_DIM + local];
    }
    qk[K_ROW + local] = projection[base + HIDDEN + kvHead * HEAD_DIM + local];
    vcur[local] = f32(f16(projection[base + HIDDEN + KV_WIDTH + kvHead * HEAD_DIM + local]));
  }
  workgroupBarrier();
  attnCore(local, kvHead, position, queryIndex);
}
`,bindings:X,workgroupSize:J}}function o7(L,$){return[L.kvHeads,$,1]}var g1=64,p0=68;function e7(L,$,J){return J*$*L.heads*p0*4}function p9(L,$,J,Z){let X=i7("attnCoreFlashKernel",L,$);if($!==L.headDim)throw Error(`attnCoreFlashKernel: the split-position core is one thread per head dim, so workgroup ${$} must be ${L.headDim}`);if(g1!==$)throw Error(`attnCoreFlashKernel: a ${g1}-position tile needs ${g1} threads, the workgroup has ${$}`);if(!Number.isInteger(J)||J<1)throw Error(`attnCoreFlashKernel: splits must be a positive integer, got ${J}`);if(!Number.isInteger(Z)||Z<1)throw Error(`attnCoreFlashKernel: maxTotal must be a positive integer, got ${Z}`);if(Z>L.maxPositions)throw Error(`attnCoreFlashKernel: maxTotal ${Z} is past the ${L.maxPositions}-position cache`);return X}function d9(L,$){return Math.ceil(Math.ceil($/L)/g1)}function L2(L,$,J,Z,X){let Y=p9(L,J,Z,X),Q=[{binding:c.blob,name:"blob",access:"read"},{binding:c.state,name:"state",access:"read"},{binding:c.keyCache,name:"key_cache",access:"read_write"},{binding:c.valueCache,name:"value_cache",access:"read_write"},{binding:c.projection,name:"projection",access:"read"},{binding:c.flash,name:"flash",access:"read_write"}];return{code:`${O1}${F1}
const HIDDEN: u32 = ${L.hidden}u;
const HEADS: u32 = ${L.heads}u;
const WG: u32 = ${J}u;
const EPSILON: f32 = ${L.epsilon};
const PROJ_ROWS: u32 = ${L.hidden+2*L.kvHeads*L.headDim}u;
const SPLITS: u32 = ${Z}u;
const TILE: u32 = ${g1}u;
const TILE_ITERS: u32 = ${d9(Z,X)}u;
const RECORD: u32 = ${p0}u;
const NEG_INF: f32 = -3.0e38;
${r7(L,$,J)}
${C1(Q,{})}

var<workgroup> ktile: array<f16, ${g1*L.headDim}>;
var<workgroup> vtile: array<f16, ${g1*L.headDim}>;
var<workgroup> qk: array<f32, ${(Y+1)*L.headDim}>;
var<workgroup> vcur: array<f32, ${L.headDim}>;
var<workgroup> reduce: array<f32, ${J}>;
var<workgroup> sc: array<f32, ${Y*g1}>;
${t7()}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let kvHead = wid.x / SPLITS;
  let s = wid.x % SPLITS;
  let queryIndex = wid.y;
  let position = state.position + queryIndex;
  let total = position + 1u;
  let blockLen = (total + SPLITS - 1u) / SPLITS;
  let start = s * blockLen;
  let stop = min(start + blockLen, total);
  // The block that holds this token's own position. Under the registered S rule it is the last
  // one, but a short token under a large S leaves trailing blocks empty and it is not, so it is
  // computed rather than assumed.
  let owner = (total - 1u) / blockLen;

  // Every block needs q. Only the block that owns the current position needs this token's k and v,
  // and reading them in every block would multiply the projection scratch by S. The rows the other
  // blocks skip are written to zero rather than left to the implementation: they still take the
  // norm and the rotation, because dropping them would put a barrier under a non-uniform branch,
  // and nothing reads the result.
  let base = queryIndex * PROJ_ROWS;
  for (var g = 0u; g < GROUP; g = g + 1u) {
    qk[g * HEAD_DIM + local] = projection[base + (kvHead * GROUP + g) * HEAD_DIM + local];
  }
  if (s == owner) {
    qk[K_ROW + local] = projection[base + HIDDEN + kvHead * HEAD_DIM + local];
    vcur[local] = f32(f16(projection[base + HIDDEN + KV_WIDTH + kvHead * HEAD_DIM + local]));
  } else {
    qk[K_ROW + local] = 0.0;
    vcur[local] = 0.0;
  }
  workgroupBarrier();
  headRows(local, position);
  if (s == owner) {
    key_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(qk[K_ROW + local]);
    value_cache[(kvHead * MAX_POSITIONS + position) * HEAD_DIM + local] = f16(vcur[local]);
  }

  // The running state of the online softmax. Every one of these is computed from workgroup-uniform
  // data and is identical in every lane, so it lives in registers and costs no workgroup memory --
  // except vacc, which is per dim, and thread local owns dim local in every tile.
  var m: array<f32, ${Y}>;
  var l: array<f32, ${Y}>;
  var vacc: array<f32, ${Y}>;
  var corr: array<f32, ${Y}>;
  var peak: array<f32, ${Y}>;
  for (var g = 0u; g < GROUP; g = g + 1u) { m[g] = NEG_INF; l[g] = 0.0; vacc[g] = 0.0; }

  for (var t = 0u; t < TILE_ITERS; t = t + 1u) {
    let tileStart = start + t * TILE;
    // TILE_ITERS is baked from the deepest total the run reaches, because one encoded command
    // buffer serves every token of a run that sweeps depth. At shallower positions the trailing
    // tiles are empty, and an empty tile must cost a branch rather than two full workgroup
    // reductions per q head -- otherwise the core's price would be set by the end of the run
    // instead of by the position it is at. The test is workgroup-uniform: start and stop come from
    // workgroup_id and a read-only storage load, and t is a uniform loop counter, so every barrier
    // below stays in uniform control flow.
    if (tileStart >= stop) { continue; }
    let len = min(TILE, stop - tileStart);

    // No barrier in this loop, so a runtime bound is legal.
    for (var p = 0u; p < len; p = p + 1u) {
      let row = (kvHead * MAX_POSITIONS + tileStart + p) * HEAD_DIM + local;
      ktile[p * HEAD_DIM + local] = key_cache[row];
      vtile[p * HEAD_DIM + local] = value_cache[row];
    }
    workgroupBarrier();

    // Thread local scores position local of the tile against all GROUP q heads in one pass
    // over its staged k row, in ascending d -- the same term order as the core this replaces.
    var sv: array<f32, ${Y}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = NEG_INF; }
    if (local < len) {
      var a: array<f32, ${Y}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = 0.0; }
      for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
        let kd = f32(ktile[local * HEAD_DIM + d]);
        for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = a[g] + qk[g * HEAD_DIM + d] * kd; }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = a[g] * SCALE; }
    }

    for (var g = 0u; g < GROUP; g = g + 1u) {
      reduce[local] = sv[g];
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
        workgroupBarrier();
      }
      peak[g] = reduce[0];
      workgroupBarrier();
    }

    // The online update. An empty tile leaves peak at NEG_INF, so corr is exp(0) = 1 and the
    // denominator gains nothing; an empty BLOCK never leaves NEG_INF, and the merge prices its
    // weight at exp(NEG_INF - M) = 0 against any real maximum.
    for (var g = 0u; g < GROUP; g = g + 1u) {
      let next = max(m[g], peak[g]);
      corr[g] = exp(m[g] - next);
      var e: f32 = 0.0;
      if (local < len) { e = exp(sv[g] - next); }
      sc[g * TILE + local] = e;
      reduce[local] = e;
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
        workgroupBarrier();
      }
      l[g] = l[g] * corr[g] + reduce[0];
      m[g] = next;
      workgroupBarrier();
    }

    // Thread local owns dim local and walks the tile's positions in index order, so the sum is
    // fixed and the staged V row it reads is contiguous across the workgroup.
    var w: array<f32, ${Y}>;
    for (var g = 0u; g < GROUP; g = g + 1u) { w[g] = 0.0; }
    for (var p = 0u; p < len; p = p + 1u) {
      let vd = f32(vtile[p * HEAD_DIM + local]);
      for (var g = 0u; g < GROUP; g = g + 1u) { w[g] = w[g] + sc[g * TILE + p] * vd; }
    }
    for (var g = 0u; g < GROUP; g = g + 1u) { vacc[g] = vacc[g] * corr[g] + w[g]; }
    workgroupBarrier();
  }

  for (var g = 0u; g < GROUP; g = g + 1u) {
    let rec = ((queryIndex * SPLITS + s) * HEADS + kvHead * GROUP + g) * RECORD;
    flash[rec + local] = vacc[g];
    if (local == 0u) {
      flash[rec + HEAD_DIM] = m[g];
      flash[rec + HEAD_DIM + 1u] = l[g];
    }
  }
}
`,bindings:Q,workgroupSize:J}}function $2(L,$,J){return[L.kvHeads*$,J,1]}function J2(L,$,J){if($!==L.headDim)throw Error(`attnMergeKernel: one thread per head dim, so workgroup ${$} must be ${L.headDim}`);if(!Number.isInteger(J)||J<1)throw Error(`attnMergeKernel: splits must be a positive integer, got ${J}`);let Z=[{binding:c.destination,name:"destination",access:"read_write"},{binding:c.flash,name:"flash",access:"read"}];return{code:`${O1}
const HEADS: u32 = ${L.heads}u;
const HEAD_DIM: u32 = ${L.headDim}u;
const WG: u32 = ${$}u;
const SPLITS: u32 = ${J}u;
const RECORD: u32 = ${p0}u;
const NEG_INF: f32 = -3.0e38;

${C1(Z,{})}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let head = wid.x;
  let queryIndex = wid.y;
  let first = (queryIndex * SPLITS * HEADS + head) * RECORD;
  var peak: f32 = NEG_INF;
  for (var s = 0u; s < SPLITS; s = s + 1u) {
    peak = max(peak, flash[first + s * HEADS * RECORD + HEAD_DIM]);
  }
  var num: f32 = 0.0;
  var den: f32 = 0.0;
  for (var s = 0u; s < SPLITS; s = s + 1u) {
    let rec = first + s * HEADS * RECORD;
    let weight = exp(flash[rec + HEAD_DIM] - peak);
    num = num + weight * flash[rec + local];
    den = den + weight * flash[rec + HEAD_DIM + 1u];
  }
  destination[(queryIndex * HEADS + head) * HEAD_DIM + local] = f16(num / den);
}
`,bindings:Z,workgroupSize:$}}function Z2(L,$){return[L.heads,$,1]}function X2(L,$,J){m9("mlpFusedKernel",L,J,L.ffn);let{workgroupSize:Z,subgroupSize:X,slices:Y}=J,Q=L.ffn/Y;if(Q%L.quantBlock!==0)throw Error(`mlpFusedKernel: ${Q} rows per slice is not a whole number of ${L.quantBlock}-wide blocks`);let H=u9(J);return{code:`${O1}${F1}${g0}
${h9(L,J)}const FFN: u32 = ${L.ffn}u;
const ROWS_PER_SLICE: u32 = ${Q}u;
const SLICE_ITERS: u32 = ${Q/(Z/X)}u;

${C1(H,{})}

var<workgroup> stage: array<f32, ${L.hidden}>;
var<workgroup> reduce: array<f32, ${Z}>;
var<workgroup> hidden_slice: array<f32, ${Q}>;
${g9(L,$.norm,J.foldSlices)}${f9($.gateUpQuant,$.gateUpScales,L.hidden/L.quantBlock,X)}${k9($.downKQuant,$.downKScales,Q/L.quantBlock,"hidden_slice")}
@compute @workgroup_size(WG)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_index) local: u32,
  @builtin(subgroup_invocation_id) lane: u32,
) {
  let group = wid.x;
  stageBlock(local, group);
  let sub = local / SG;
  let rowBase = group * ROWS_PER_SLICE;
  for (var i = 0u; i < SLICE_ITERS; i = i + 1u) {
    let r = sub + i * ROWS_AT_ONCE;
    let gate = rowSum(rowBase + r, lane);
    let up = rowSum(FFN + rowBase + r, lane);
    if (lane == 0u) { hidden_slice[r] = f32(f16((gate / (1.0 + exp(-gate))) * up)); }
  }
  workgroupBarrier();
  project(local, group);
}
`,bindings:H,workgroupSize:Z}}function Y2(L,$,J,Z){if($%J!==0)throw Error(`foldKernel: workgroup ${$} is not a multiple of ${J} threads per row`);if(Z<1)throw Error(`foldKernel: nothing to fold, got ${Z} slices`);let X=$/J,Y=[{binding:c.residualOut,name:"residual_out",access:"read_write"},{binding:c.partialsIn,name:"partials_in",access:"read"}];return{code:`${O1}
const HIDDEN: u32 = ${L.hidden}u;
const WG: u32 = ${$}u;
const LANES: u32 = ${J}u;
const ROWS_PER_WG: u32 = ${X}u;
const FOLD_SLICES: u32 = ${Z}u;

${C1(Y,{})}

var<workgroup> part: array<f32, ${$}>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  let slot = local % ROWS_PER_WG;
  let lane = local / ROWS_PER_WG;
  let row = wid.x * ROWS_PER_WG + slot;
  var acc: f32 = 0.0;
  if (row < HIDDEN) {
    for (var s = lane; s < FOLD_SLICES; s = s + LANES) { acc = acc + partials_in[s * HIDDEN + row]; }
  }
  part[local] = acc;
  workgroupBarrier();
  if (lane == 0u && row < HIDDEN) {
    var sum: f32 = residual_out[row];
    for (var l = 0u; l < LANES; l = l + 1u) { sum = sum + part[l * ROWS_PER_WG + slot]; }
    residual_out[row] = sum;
  }
}
`,bindings:Y,workgroupSize:$}}function Q2(L){return[L,1,1]}function H2(L,$,J){return[Math.ceil(L.hidden/($/J)),1,1]}var D0=0,Y0="carried";function j2(L){if(![0,2,4,8].includes(L))throw Error("Unregistered padding");D0=L}function K2(L){if(!["carried","subgroup","vec4","both"].includes(L))throw Error("Unregistered flash variant");Y0=L}function o(L,$,J){if(L.split($).length!==2)throw Error("Flash source seam drift: "+$.slice(0,70));return L.replace($,J)}function V2(...L){let $=L2(...L),J=$.code,Z=L[0],X=Z.headDim+D0;if(D0)J=o(J,`var<workgroup> ktile: array<f16, ${64*Z.headDim}>;`,`var<workgroup> ktile: array<f16, ${64*X}>;`),J=o(J,"ktile[p * HEAD_DIM + local]",`ktile[p * ${X}u + local]`),J=o(J,"ktile[local * HEAD_DIM + d]",`ktile[local * ${X}u + d]`);if(Y0==="subgroup"||Y0==="both")J=o(J,`    reduce[local] = 0.0;
    workgroupBarrier();
    if (local < HEAD_DIM) { reduce[local] = qk[off + local] * qk[off + local]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
      if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
      workgroupBarrier();
    }
    let inv = inverseSqrt(reduce[0] / f32(HEAD_DIM) + EPSILON);`,`    let square = qk[off + local] * qk[off + local];
    let subgroupSquare = subgroupAdd(square);
    if (local % 32u == 0u) { reduce[local / 32u] = subgroupSquare; }
    workgroupBarrier();
    let inv = inverseSqrt((reduce[0] + reduce[1]) / f32(HEAD_DIM) + EPSILON);`),J=o(J,`      reduce[local] = sv[g];
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = max(reduce[local], reduce[local + stride]); }
        workgroupBarrier();
      }
      peak[g] = reduce[0];`,`      let subgroupPeak = subgroupMax(sv[g]);
      if (local % 32u == 0u) { reduce[local / 32u] = subgroupPeak; }
      workgroupBarrier();
      peak[g] = max(reduce[0], reduce[1]);`),J=o(J,`      reduce[local] = e;
      workgroupBarrier();
      for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
        workgroupBarrier();
      }
      l[g] = l[g] * corr[g] + reduce[0];`,`      let subgroupDenom = subgroupAdd(e);
      if (local % 32u == 0u) { reduce[local / 32u] = subgroupDenom; }
      workgroupBarrier();
      l[g] = l[g] * corr[g] + (reduce[0] + reduce[1]);`);if(Y0==="vec4"||Y0==="both"){let Y=Z.heads/Z.kvHeads,Q=D0?`${X}u`:"HEAD_DIM";J=o(J,`      var a: array<f32, ${Y}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = 0.0; }
      for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
        let kd = f32(ktile[local * ${Q} + d]);
        for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = a[g] + qk[g * HEAD_DIM + d] * kd; }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = a[g] * SCALE; }`,`      var a: array<vec4<f32>, ${Y}>;
      for (var g = 0u; g < GROUP; g = g + 1u) { a[g] = vec4<f32>(0.0); }
      for (var d = 0u; d < HEAD_DIM; d = d + 4u) {
        let b = local * ${Q} + d;
        let kd = vec4<f32>(f32(ktile[b]), f32(ktile[b+1u]), f32(ktile[b+2u]), f32(ktile[b+3u]));
        for (var g = 0u; g < GROUP; g = g + 1u) {
          let q = g * HEAD_DIM + d;
          a[g] = a[g] + vec4<f32>(qk[q],qk[q+1u],qk[q+2u],qk[q+3u]) * kd;
        }
      }
      for (var g = 0u; g < GROUP; g = g + 1u) { sv[g] = ((a[g].x+a[g].y)+(a[g].z+a[g].w)) * SCALE; }`)}return{...$,code:J}}var Q0="carried";function U2(L){if(!["carried","subgroup","packed","both"].includes(L))throw Error("Unknown MLP mode");Q0=L}function W2(...L){let $=L[2].slices===160,J=X2(L[0],L[1],$?{...L[2],slices:80}:L[2]),Z=J.code;if($){let X=L[2].workgroupSize;Z=o(Z,"const SLICES: u32 = 80u;","const SLICES: u32 = 160u;"),Z=o(Z,"const ROWS_PER_SLICE: u32 = 32u;","const ROWS_PER_SLICE: u32 = 16u;"),Z=o(Z,`const SLICE_ITERS: u32 = ${32/(X/32)}u;`,`const SLICE_ITERS: u32 = ${16/(X/32)}u;`),Z=o(Z,"var<workgroup> hidden_slice: array<f32, 32>;","var<workgroup> hidden_slice: array<f32, 16>;"),Z=o(Z,"KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u","KQUANT + ((group / 2u) * HIDDEN + row) * 4u + (group % 2u) * 2u"),Z=o(Z,"KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b","KSCALES + (group / 2u) * HIDDEN + row");let Y=Z.indexOf("fn project("),Q=Z.indexOf("@compute",Y),H=Z.slice(Y,Q),j=0;if(H=H.replace(/      part = part \+ dot8\(blob\[base \+ [23]u\], 8\.0,[\s\S]*?\);/g,()=>{return j++,""}),j!==2)throw Error("Half-block source seam drift");Z=Z.slice(0,Y)+H+Z.slice(Q)}if(!$&&L[2].slices!==80){if(![20,40].includes(L[2].slices))throw Error("Unregistered MLP slices");Z=o(Z,"KQUANT + (sliceBase + row) * WORDS_PER_ROW_SLICE + b * 4u","KQUANT + ((group * BLOCKS_PER_SLICE + b) * HIDDEN + row) * 4u"),Z=o(Z,"KSCALES + (sliceBase + row) * BLOCKS_PER_SLICE + b","KSCALES + (group * BLOCKS_PER_SLICE + b) * HIDDEN + row")}if(Q0==="subgroup"||Q0==="both")Z=o(Z,"  reduce[local] = sum;",`  let sgSum = subgroupAdd(sum);
  if (local % SG == 0u) { reduce[local / SG] = sgSum; }`),Z=o(Z,`  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(HIDDEN) + EPSILON);`,`  var squareSum = 0.0;
  for (var subgroup = 0u; subgroup < WG / SG; subgroup++) { squareSum += reduce[subgroup]; }
  let inv = inverseSqrt(squareSum / f32(HIDDEN) + EPSILON);`);if(Q0==="packed"||Q0==="both"){let X=0;if(Z=Z.replace(/part = part \+ dot8\((blob\[[^\]]+\]), 8\.0,\s*(vec4<f32>\([^;]+?)\);/g,(Y,Q,H)=>{X++;let j=H.replaceAll("vec4<f32>","vec4<f16>").replace(/(stage|hidden_slice)\[[^\]]+\]/g,(V)=>`f16(${V})`);return`part = part + f32(dot8h(${Q}, f16(8.0), ${j}));`}),X!==8)throw Error("MLP dot source seam drift "+X);Z+=`
`+X0}return{...J,code:Z}}function G2(L,$){let J=v7(...L);if(!$||L[0]==="matvec_residual")return J;let Z=J.code,X=`  reduce[local] = sum;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { reduce[local] = reduce[local] + reduce[local + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(reduce[0] / f32(COLS) + EPSILON);`;if(Z.split(X).length!==2)throw Error("Matvec RMS source seam drift");return Z=Z.replace(X,`  let sgSum = subgroupAdd(sum);
  if (local % SG == 0u) { reduce[local / SG] = sgSum; }
  workgroupBarrier();
  var squareSum = 0.0;
  for (var subgroup = 0u; subgroup < WG / SG; subgroup++) { squareSum += reduce[subgroup]; }
  let inv = inverseSqrt(squareSum / f32(COLS) + EPSILON);`),{...J,code:Z}}var q2=8;function X1(L,$){if($.offset%256!==0)throw Error(`runtime: section offset ${$.offset} is not 256-aligned`);return{buffer:L,offset:$.offset,size:$.bytes}}function c9(L,$,J){return L.createBindGroupLayout({label:J,entries:$.bindings.map((Z)=>({binding:Z.binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:Z.access==="read"?"read-only-storage":"storage"}}))})}function A2(L,$,J){let Z=c9(L,$,`${J}.layout`),X=L.createShaderModule({code:$.code,label:J});return{pipeline:L.createComputePipeline({label:J,layout:L.createPipelineLayout({bindGroupLayouts:[Z]}),compute:{module:X,entryPoint:"main"}}),layout:Z}}function a9(L,$){return{hidden:L.hidden,ffn:L.ffn,vocab:L.vocab,heads:L.heads,kvHeads:L.kvHeads,headDim:L.headDim,convTaps:L.convCache,quantBlock:L.quantBlock,epsilon:e1.normEps,maxPositions:$.maxPositions,maxQueries:$.maxQueries}}class O2{device;entries=new Map;constructor(L){this.device=L}get(L,$){let J=this.entries.get(L);if(J!==void 0)return J;let Z=$(),{pipeline:X,layout:Y}=A2(this.device,Z,L),Q={pipeline:X,layout:Y,kernel:Z};return this.entries.set(L,Q),Q}get size(){return this.entries.size}}function F2(L,$,J,Z,X,Y,Q){if(Y.length===0)throw Error("runtime: no arms to build");let H=a9($,Q),{maxQueries:j,maxPositions:V}=Q,G=$.layerTypes.filter((K)=>K==="attention").length,I=$.layerTypes.length-G,D=(K)=>{let W=X.get(K);if(W===void 0)throw Error(`runtime: weight section ${K} missing`);return W},R=(K)=>{let W=D(K);if(W.offset%4!==0)throw Error(`runtime: section ${K} at ${W.offset} is not word aligned`);return W.offset/4},z=(K)=>{let W=D(`${K}.norm`),b=D(`${K}.q_norm`),x=D(`${K}.k_norm`);if(b.offset!==W.offset+W.bytes||x.offset!==b.offset+b.bytes)throw Error(`runtime: ${K} norm/q_norm/k_norm are not contiguous`);return{offset:W.offset,bytes:W.bytes+b.bytes+x.bytes}},T=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,N=(K,W)=>L.createBuffer({size:K,usage:T,label:W}),y=j*$.hidden*s.residualBytes,E=Q.maxSlices*$.hidden*s.residualBytes,t=j*3*$.hidden*s.hiddenBytes,e=j*Math.max($.hidden,$.ffn)*s.hiddenBytes,k=j*($.hidden+2*$.kvHeads*$.headDim)*4,h=$.kvHeads*V*$.headDim*s.kvCacheBytes,L1=$.hidden*$.convCache*s.convCacheBytes,U1=j*$.vocab*s.logitsBytes,w1=Y.reduce((K,W)=>Math.max(K,W.options.flash.blocks),1),R1=e7(H,w1,j),B=Q.samplePartials;if(h%256!==0||L1%256!==0)throw Error("runtime: per-layer cache stride is not 256-aligned");let I1=N(y,"residual_a"),i=N(y,"residual_b"),n=N(y,"normalized"),$1=N(E,"deferred_a"),j1=N(E,"deferred_b"),x1=N(t,"proj"),H1=N(e,"mid"),K1=N(k,"qkv_proj"),W1=N(G*h,"key_cache"),B1=N(G*h,"value_cache"),G1=N(I*L1,"conv_cache"),C=N(U1,"logits"),U=N(q2*4,"state"),A=N(V*4,"tokens"),M=N(V*4,"sampled"),P=N(B*2*4,"partials"),S=N(R1,"flash"),O=(K)=>({buffer:K,offset:0,size:K.size}),w=(K,W,b)=>({buffer:K,offset:W*b,size:b}),u=(K,W,b,x)=>{if(K===0)return W;if(K===1)return b;throw Error(`runtime: ${x} parity ${K} is not 0 or 1`)},v=(K)=>O(u(K,I1,i,"residual")),J1=(K)=>O(u(K,$1,j1,"partials")),q1=2*$.ffn*$.hidden/2,_1=$.ffn*$.hidden/2,U0=(K)=>{if(!Y.some((d)=>{let Y1=J0(d.key).mlpSoA;return Y1===K||Y1==="both"}))return null;let W=K==="gate"?q1:_1,b=N(W*$.layerTypes.length,"mlp_"+K+"_soa"),x=T7($.hidden,$.ffn,K),m=A2(L,x,"repack_"+K),p=L.createCommandEncoder(),l=p.beginComputePass();l.setPipeline(m.pipeline);for(let d=0;d<$.layerTypes.length;d++){let Y1=D(`L${d}.mlp.${K==="gate"?"gate_up.quant":"down.kquant"}`);if(Y1.bytes!==W)throw Error("SoA source size mismatch");let p1=L.createBindGroup({layout:m.layout,entries:[{binding:0,resource:X1(J,Y1)},{binding:1,resource:w(b,d,W)}]});l.setBindGroup(0,p1),l.dispatchWorkgroups(Math.ceil(W/4/256))}return l.end(),L.queue.submit([p.finish()]),b},W0=U0("gate"),G0=U0("down"),V1=new O2(L),l2=(K,W,b,x)=>{let m=W.map((p)=>{let l=b[p.name];if(l===void 0)throw Error(`runtime: ${x} has no resource for ${p.name}`);return{binding:p.binding,resource:l}});return L.createBindGroup({layout:K,entries:m,label:x})},p2=(K)=>({norm:R(`L${K}.mlp.gate_up.norm`),gateUpQuant:R(`L${K}.mlp.gate_up.quant`),gateUpScales:R(`L${K}.mlp.gate_up.scales`),downKQuant:R(`L${K}.mlp.down.kquant`),downKScales:R(`L${K}.mlp.down.kscales`)}),v0=(K)=>({qNorm:R(`L${K}.attn.qkv.q_norm`),kNorm:R(`L${K}.attn.qkv.k_norm`),cos:R("rope.cos_ext"),sin:R("rope.sin_ext")}),d2=(K,W)=>{if(W.fusion.slices!==K.slices)throw Error(`runtime: ${W.name} publishes ${W.fusion.slices} slices, the kernel is built for ${K.slices}`);if(K.slices>Q.maxSlices)throw Error(`runtime: ${W.name} publishes ${K.slices} slices into a ${Q.maxSlices}-row partial buffer`);return{workgroupSize:K.workgroupSize,subgroupSize:K.subgroupSize,slices:K.slices,foldSlices:W.fusion.foldSlices}},r1=(K)=>{if(K.weight===null)throw Error(`runtime: ${K.name} is a matmul with no weight`);return K.weight},c2=(K)=>{let W=J0(K.key);j2(W.padding),K2(W.flashMode),U2(W.mlpMode??"carried");let b=f0(F0($,K.options),W,$,K.options.position,K.options.queries),x=K.options.queries,m=K.options.flash.blocks,p=Q.flashMaxTotalPositions,l=[],d=0,Y1=0,p1=!1,A1=(q,_,F,r)=>{if(_.kernel.workgroupSize!==q.workgroupSize)throw Error(`runtime: ${q.name} is planned at workgroup size ${q.workgroupSize}, its kernel is ${_.kernel.workgroupSize}`);let[Z1,E1,f1]=r(x);if(Z1*E1*f1!==q.workgroups)throw Error(`runtime: ${q.name} is planned at ${q.workgroups} workgroups, the runtime encodes ${Z1*E1*f1}`);l.push({plan:q,pipeline:_.pipeline,bindGroup:l2(_.layout,_.kernel.bindings,F,`${K.key}.${q.name}`),workgroups:r})},i1=(q,_,F,r,Z1,E1,f1,b0)=>{if(q.site===null||q.geometry===null)throw Error(`runtime: ${q.name} is a matvec, so the plan must name the site and the geometry it runs at`);let y0=z7(q.geometry,Q.matvecMaxCols),{workgroupSize:r2,subgroupSize:i2,rowsPerSubgroup:t2,subgroupsPerRow:s2}=y0,o2=`w${r2}s${i2}r${t2}c${s2}`,L7=W.matvec[q.site]?.variant??"carried",$7=q.kind==="conv_proj_core",S0=m0(W,q),e2=`${_}_${F}x${r}${Z1?"_zp":""}_${o2}_${L7}_n${W.normSubgroup??!1}_sw${W.matvecStageSwizzle??!1}_pv${W.matvecPrivate??!1}_once${S0}_conv${$7}_mask${W.maskMatvec??!1}_magic${W.magicMatvec??!1}`,L9=V1.get(e2,()=>h0(u0(p7(l7(u7(l0(G2([_,F,r,Z1,y0,H,E1,L7],W.normSubgroup??!1),W.matvecStageSwizzle??!1),_,E1,W.matvecPrivate??!1),S0),$7),W.maskMatvec??!1),W.magicMatvec??!1)),w0={weights:X1(J,D(`${f1}.quant`)),scales:X1(J,D(`${f1}.scales`)),state:O(U),partials:O(P),...b0};if(S0)w0.residual=O(n);if(Z1)w0.zero_points=X1(J,D(`${f1}.zero_points`));A1(q,L9,w0,($9)=>P7(_,F,H,y0,$9))},n2=(q,_,F,r,Z1)=>{let E1=d2(F,q),f1=V1.get(_,()=>r(E1)),b0={blob:O(J),weights:O(J),projection:O(n),residual_in:v(q.fusion.residualIn),residual_out:v(q.fusion.residualOut),partials_in:J1(q.fusion.partialsIn),partials_out:J1(q.fusion.partialsOut),state:O(U),...Z1};A1(q,f1,b0,()=>Q2(F.slices)),p1=!0};for(let q of b){let _=q.fusion;switch(q.kind){case"input_norm":{let F=q.name.replace(/\.input_norm$/,""),r=F.endsWith(".mlp.fused")?F.replace(/\.fused$/,".gate_up.norm"):F+".norm",Z1=F==="head"?Q.headPreScale:1,E1=V1.get("input_norm_"+q.workgroupSize+"_"+Z1,()=>d7(q.workgroupSize,Z1));A1(q,E1,{residual:v(_.residualIn),gamma:X1(J,D(r)),projection:O(n)},()=>[1,1,1]);break}case"embed":{let F=V1.get("embed",()=>b7(H,Q.embedWorkgroup));A1(q,F,{weights:X1(J,D("embed.quant")),scales:X1(J,D("embed.scales")),zero_points:X1(J,D("embed.zero_points")),residual:v(_.residualOut),state:O(U),tokens:O(A)},(r)=>y7(r));break}case"conv_proj_core":{let F=r1(q);i1(q,"norm_matvec",F.rows,F.cols,F.zeroPoints,1,q.name,{gamma:X1(J,D(q.name+".norm")),residual:v(_.residualIn),destination:O(H1),blob:X1(J,D(q.name.replace(/\.in_proj$/,".core.taps"))),cache:w(G1,Y1,L1)}),Y1++;break}case"norm_matvec":{let F=r1(q);i1(q,"norm_matvec",F.rows,F.cols,F.zeroPoints,1,q.name,{gamma:X1(J,D(`${q.name}.norm`)),residual:v(_.residualIn),destination:O(x1)});break}case"conv_core":{let F=V1.get("conv_core",()=>S7(H,Q.convWorkgroup));A1(q,F,{gamma:X1(J,D(`${q.name}.taps`)),source:O(x1),destination:O(H1),state:O(U),cache:w(G1,Y1,L1)},()=>w7(H,Q.convWorkgroup)),Y1+=1;break}case"matvec_residual":{let F=r1(q);i1(q,"matvec_residual",F.rows,F.cols,F.zeroPoints,1,q.name,{residual:v(_.residualOut),source:O(H1)});break}case"norm_matvec_swiglu":{let F=r1(q);i1(q,"norm_swiglu",F.rows,F.cols,F.zeroPoints,1,q.name,{gamma:X1(J,D(`${q.name}.norm`)),residual:v(_.residualIn),destination:O(H1)});break}case"attn_proj":{let F=r1(q);i1(q,"norm_projection",F.rows,F.cols,F.zeroPoints,1,q.name,{gamma:X1(J,D(`${q.name}.norm`)),residual:v(_.residualIn),projection:O(K1)});break}case"attn_core_qkv":{let F=q.layer,r=V1.get(`attn_core_qkv_L${F}`,()=>s7(H,v0(F),Q.attnCoreWorkgroup));A1(q,r,{blob:O(J),state:O(U),key_cache:w(W1,d,h),value_cache:w(B1,d,h),projection:O(K1),destination:O(H1)},(Z1)=>o7(H,Z1)),d+=1;break}case"attn_core_flash":{let F=q.layer,r=V1.get(`attn_core_flash_L${F}_s${m}_t${p}_p${W.padding}_${W.flashMode}_online${W.flashOnline??!1}`,()=>W.flashOnline?g7(H,v0(F),m):V2(H,v0(F),Q.flashCoreWorkgroup,m,p));A1(q,r,{blob:O(J),state:O(U),key_cache:w(W1,d,h),value_cache:w(B1,d,h),projection:O(K1),flash:O(S)},(Z1)=>$2(H,m,Z1)),d+=1;break}case"attn_merge":{let F=V1.get(`attn_merge_s${m}_sg${W.mergeSubgroup??!1}`,()=>k7(J2(H,Q.mergeWorkgroup,m),m,W.mergeSubgroup??!1));A1(q,F,{flash:O(S),destination:O(H1)},(r)=>Z2(H,r));break}case"mlp_fused":{let F=q.layer;n2(q,`mlp_fused_${W.key}_L${F}_f${_.foldSlices}_${W.mlpMode??"carried"}_s${W.mlpSlices??80}_w${W.mlpWorkgroup??256}`,{...Q.mlpFused,slices:W.mlpSlices??Q.mlpFused.slices,workgroupSize:W.mlpWorkgroup??Q.mlpFused.workgroupSize},(r)=>c7(W2(H,p2(F),r),W),{...(W.mlpSoA==="gate"||W.mlpSoA==="both")&&W0?{weights:w(W0,F,q1)}:{},...(W.mlpSoA==="down"||W.mlpSoA==="both")&&G0?{source:w(G0,F,_1)}:{}});break}case"fold":{if(_.residualIn!==_.residualOut||_.partialsIn!==_.partialsOut)throw Error(`runtime: ${q.name} is in place, so its parities must match`);let F=V1.get(`fold_${W.key}_${_.foldSlices}`,()=>a7(Y2(H,Q.fold.workgroupSize,W.foldLanes??Q.fold.threadsPerRow,_.foldSlices),W));A1(q,F,{residual_out:v(_.residualOut),partials_in:J1(_.partialsIn)},()=>H2(H,Q.fold.workgroupSize,W.foldLanes??Q.fold.threadsPerRow));break}case"norm_head":{let F=r1(q);i1(q,"norm_head",F.rows,F.cols,F.zeroPoints,Q.headPreScale,"head",{gamma:X1(J,D("head.norm")),residual:v(_.residualIn),logits:O(C)});break}case"sample_partial":{let F=V1.get(`sample_partial_${B}`,()=>x7(H,Q.sampleWorkgroup,B));A1(q,F,{state:O(U),partials:O(P),logits:O(C)},()=>m7(B));break}case"sample_final":{let F=V1.get(`sample_final_${B}`,()=>_7(H,Q.sampleWorkgroup,B));A1(q,F,{state:O(U),tokens:O(A),partials:O(P),sampled:O(M)},()=>f7());break}}}if(l.length!==b.length)throw Error(`runtime: arm ${K.key} encoded ${l.length} of ${b.length} planned dispatches`);if(p1&&x!==1)throw Error(`runtime: arm ${K.key} is fused, which is a single-query decode specialisation, and plans ${x} queries`);return{key:K.key,plan:b,steps:l,fused:p1}},m1=new Map;for(let K of Y){if(m1.has(K.key))throw Error(`runtime: arm ${K.key} built twice`);m1.set(K.key,c2(K))}let s0=m1.get(Y[0]?.key??"");if(s0===void 0)throw Error("runtime: the first arm did not build");let N1=s0,o0=[...W0?[W0]:[],...G0?[G0]:[],n,I1,i,$1,j1,x1,H1,K1,W1,B1,G1,C,U,A,M,P,S],a2=o0.reduce((K,W)=>K+W.size,Z),e0=0;for(let K of m1.values())e0+=K.steps.length;let z1=new Uint32Array(q2),q0=(K)=>{if(K!==1&&N1.fused)throw Error(`runtime: arm ${N1.key} is fused and encodes one query, not ${K}`);return N1.steps};return{shape:$,armKeys:[...m1.keys()],get arm(){return N1.key},get steps(){return N1.steps},get dispatchCount(){return N1.steps.length},get tokensPerPass(){return J0(N1.key).tokensPerPass??1},pipelineCount:V1.size,bindGroupCount:e0,weightBytes:Z,liveGpuBytes:a2,buffers:{state:U,tokens:A,sampled:M,logits:C,residual:I1,residualB:i,deferredA:$1,deferredB:j1,partials:P,proj:x1,qkvProj:K1,flash:S,mid:H1},setArm(K){let W=m1.get(K);if(W===void 0)throw Error(`runtime: no arm ${K}`);N1=W},dispatchCountOf(K){let W=m1.get(K);if(W===void 0)throw Error(`runtime: no arm ${K}`);return W.steps.length},planOf(K){let W=m1.get(K);if(W===void 0)throw Error(`runtime: no arm ${K}`);return W.plan},writeState(K){z1[0]=K.position,z1[1]=K.step,z1[2]=K.queries,z1[3]=K.sampleMode,z1[4]=K.rng,z1[5]=K.freerun,z1[6]=K.advance,z1[7]=0,L.queue.writeBuffer(U,0,z1)},writeTokens(K,W){L.queue.writeBuffer(A,W*4,K)},encodeChunk(K,W,b){let x=K.beginComputePass(b?{timestampWrites:b}:{});for(let m=0;m<W;m++)for(let p of q0(1)){x.setPipeline(p.pipeline),x.setBindGroup(0,p.bindGroup);let[l,d,Y1]=p.workgroups(1);x.dispatchWorkgroups(l,d,Y1)}x.end()},encodeToken(K,W,b){let x=b===null?K.beginComputePass():K.beginComputePass({timestampWrites:b});for(let m of q0(W)){x.setPipeline(m.pipeline),x.setBindGroup(0,m.bindGroup);let[p,l,d]=m.workgroups(W);x.dispatchWorkgroups(p,l,d)}x.end()},assertReach(K){if(!Number.isInteger(K)||K<1)throw Error(`runtime: ${K} is not a positive total position count`);if(K>Q.flashMaxTotalPositions)throw Error(`runtime: a run reaching ${K} total positions was built for ${Q.flashMaxTotalPositions}, so the split-position core would drop every position past its last tile`);if(K>V)throw Error(`runtime: a run reaching ${K} total positions is past the ${V}-position cache`)},stepIndex(K){let W=N1.steps.findIndex((b)=>b.plan.name===K);if(W<0)throw Error(`runtime: arm ${N1.key} has no planned dispatch named ${K}`);return W},encodeOne(K,W,b){let x=q0(b)[W];if(x===void 0)throw Error(`runtime: dispatch index ${W} out of range`);let m=K.beginComputePass();m.setPipeline(x.pipeline),m.setBindGroup(0,x.bindGroup);let[p,l,d]=x.workgroups(b);m.dispatchWorkgroups(p,l,d),m.end()},encodePerDispatchPasses(K,W,b,x){let m=x;for(let p of q0(W)){let l=K.beginComputePass({timestampWrites:{querySet:b,beginningOfPassWriteIndex:m,endOfPassWriteIndex:m+1}});l.setPipeline(p.pipeline),l.setBindGroup(0,p.bindGroup);let[d,Y1,p1]=p.workgroups(W);l.dispatchWorkgroups(d,Y1,p1),l.end(),m+=2}return m-x},clearCaches(){let K=L.createCommandEncoder();K.clearBuffer(W1),K.clearBuffer(B1),K.clearBuffer(G1),K.clearBuffer(M),L.queue.submit([K.finish()])},clearResidual(){let K=L.createCommandEncoder();K.clearBuffer(I1),K.clearBuffer(i),K.clearBuffer($1),K.clearBuffer(j1),L.queue.submit([K.finish()])},destroy(){for(let K of o0)K.destroy()}}}function C2(L,$,J){let Z=$0(L);if(g.matvec.maxCols!==g.matvecWide.maxCols)throw Error(`registeredRuntimeOptions: the two registered geometries stage ${g.matvec.maxCols} and ${g.matvecWide.maxCols} columns, but the stage width is a kernel bound and must be one number`);return{maxQueries:t1.maxQueries,maxPositions:g.maxPositions,matvecMaxCols:g.matvec.maxCols,dequantVariant:$,samplePartials:g.sample.partials,sampleWorkgroup:g.sample.workgroupSize,embedWorkgroup:g.embed.workgroupSize,convWorkgroup:g.conv.workgroupSize,attnCoreWorkgroup:f.attnSplit.coreWorkgroupSize,flashCoreWorkgroup:f.flash.coreWorkgroupSize,mergeWorkgroup:f.flash.mergeWorkgroupSize,flashMaxTotalPositions:J,headPreScale:t1.headPreScale,mlpFused:{workgroupSize:f.mlpFused.workgroupSize,subgroupSize:g.matvec.subgroupSize,slices:Z.mlp},fold:f.fold,maxSlices:_0(L)}}function M2(L){let $=new Map;for(let J of L){if($.has(J.name))throw Error(`duplicate weight section ${J.name}`);$.set(J.name,{offset:J.offset,bytes:J.bytes})}return $}function n9(L){return Array.from(new Uint8Array(L)).map(($)=>$.toString(16).padStart(2,"0")).join("")}var r9=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;async function D2(L,$,J){let Z=performance.now(),X=await fetch($);if(!X.ok)throw Error(`weights fetch ${$}: ${X.status}`);let Y=await X.arrayBuffer(),Q=performance.now()-Z;if(Y.byteLength!==J)throw Error(`weights are ${Y.byteLength} B, manifest says ${J} B`);let H=n9(await crypto.subtle.digest("SHA-256",Y)),j=L.createBuffer({size:Y.byteLength,usage:r9,mappedAtCreation:!0,label:"weights"});return new Uint8Array(j.getMappedRange()).set(new Uint8Array(Y)),j.unmap(),{buffer:j,bytes:Y.byteLength,sha256:H,fetchMs:Q}}function i9(L){return{knee:G7(L.knee),flashOptima:L.flashOptima.map(($)=>({...$}))}}function t9(L){return{layerTypes:L.shape.layerTypes,hidden:L.shape.hidden,ffn:L.shape.ffn,vocab:L.shape.vocab,heads:L.shape.heads,kvHeads:L.shape.kvHeads,headDim:L.shape.headDim,convCache:L.shape.convCache,quantBits:L.shape.quantBits,quantBlock:L.shape.quantBlock}}function s9(L,$,J,Z){let X=t9($),Y=M2($.sections),Q=i9($),H={...C2(X,$.dequantVariant,$.maxTotalPositions),maxSlices:C7()},j=F2(L.device,X,J.buffer,J.bytes,Y,F7(X,$.planPosition-$.maxQueries,$.planPosition,Q),{...H,maxQueries:$.maxQueries,maxPositions:$.maxPositions,samplePartials:$.samplePartials});return{gpu:L,runtime:j,blob:J,shape:X,sections:Y,tuning:Q,planPosition:$.planPosition,errors:Z,destroy(){j.destroy(),J.buffer.destroy()}}}function o9(L,$,J){return{features:L,limits:{...D7,maxBufferSize:$,maxStorageBufferBindingSize:J}}}function e9(L,$,J,Z){let X=Math.max(L*$*s.logitsBytes,Z);for(let Y of J)X=Math.max(X,Y.bytes);return X}function d0(L){return o9(L.extraFeatures,L.weightBytes,e9(L.shape.vocab,L.maxQueries,L.sections,L.weightBytes))}async function I2(L){let $=[],J=await M0(d0(L),L.powerPreference,L.expectedSubgroupSize),Z=await D2(J.device,L.weightsUrl,L.weightBytes),X=performance.now(),Y=null;if(await I7(J.device,$,"runtime_build",async()=>{Y=s9(J,L,Z,$),await c1(J.device)}),Y===null)throw Error(`runtime build failed: ${$.join("; ")}`);let Q=Y,H=performance.now()-X,j={env:N7(),device:J.descriptor,dispatches_per_token:Q.runtime.dispatchCount,arm_dispatches:Q.runtime.armKeys.map((V)=>Q.runtime.dispatchCountOf(V)),pipelines:Q.runtime.pipelineCount,bind_groups:Q.runtime.bindGroupCount,weight_bytes:Z.bytes,live_gpu_bytes:Q.runtime.liveGpuBytes,build_ms:H,weights_fetch_ms:Z.fetchMs,weights_sha256:Z.sha256,subgroup_min:J.descriptor.adapter.subgroupMinSize,subgroup_max:J.descriptor.adapter.subgroupMaxSize,validation_errors:[...$]};return{session:Q,ready:j}}class N2{device;slotCount;tokensPerSlot;slots=[];pending=[];chunkAt=[];ids=[];constructor(L,$,J){this.device=L;this.slotCount=$;this.tokensPerSlot=J;for(let Z=0;Z<$;Z+=1)this.slots.push(L.createBuffer({size:J*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,label:`readback${Z}`})),this.pending.push(null),this.chunkAt.push(-1)}slot(L){let $=this.slots[L%this.slotCount];if($===void 0)throw Error("readback ring index");return $}async recycle(L){let $=L%this.slotCount,J=this.pending[$];if(J!==void 0&&J!==null)await J,this.drainSlot($);return this.slot(L)}drainSlot(L){let $=this.slots[L],J=this.chunkAt[L];if($===void 0||J===void 0||J<0)return;let Z=new Uint32Array($.getMappedRange().slice(0));$.unmap();for(let X=0;X<Z.length;X+=1){let Y=Z[X];if(Y===void 0)throw Error("readback view index");this.ids[J*this.tokensPerSlot+X]=Y}this.pending[L]=null,this.chunkAt[L]=-1}start(L){let $=L%this.slotCount;this.chunkAt[$]=L,this.pending[$]=this.slot(L).mapAsync(GPUMapMode.READ)}async drain(){for(let L=0;L<this.slotCount;L+=1){let $=this.pending[L];if($!==void 0&&$!==null)await $,this.drainSlot(L)}return this.ids}destroy(){for(let L of this.slots)L.destroy()}}async function I0(L,$){let{runtime:J,gpu:Z}=L;J.assertReach($.promptLength),J.clearCaches(),J.clearResidual(),J.writeTokens($.tokenIds,0),J.writeState({position:0,step:0,queries:1,sampleMode:0,rng:$.seed,freerun:0,advance:1});let X=Z.device.createCommandEncoder();for(let Y=0;Y<$.promptLength;Y+=1)J.encodeToken(X,1,null);if($.freerun){if($.promptLength<1)throw Error("Free generation needs a nonempty prompt");X.copyBufferToBuffer(J.buffers.sampled,($.promptLength-1)*4,J.buffers.tokens,$.promptLength*4,4)}Z.device.queue.submit([X.finish()]),await c1(Z.device)}function T2(L,$){L.runtime.writeState({position:$.startPosition,step:$.startStep,queries:1,sampleMode:0,rng:$.seed,freerun:$.freerun?1:0,advance:1})}function R2(L,$){if($.ringTokens%$.readbackEvery!==0)throw Error(`ring of ${$.ringTokens} tokens does not divide by readback every ${$.readbackEvery}`);return new N2(L,$.ringTokens/$.readbackEvery,$.readbackEvery)}async function L8(L,$,J,Z){if($===null)return{perTokenMs:[],gpuBusyMs:-1,gpuSpanMs:-1};let X=await k0(L,$,J),Y=[],Q=0;for(let V=0;V<J;V+=1){let G=X[V*2],I=X[V*2+1],D=V+1<J?X[(V+1)*2]:void 0;if(G===void 0||I===void 0)throw Error("timestamp index");Q+=I-G,Y.push(((D===void 0?I:D)-G)/1e6)}let H=X[0],j=X[J*2-1];if(H===void 0||j===void 0)throw Error("timestamp span index");return{perTokenMs:Y,gpuBusyMs:Q/1e6,gpuSpanMs:(j-H)/1e6}}async function B2(L,$){if(L.runtime.tokensPerPass>1)return $8(L,$);let{runtime:J,gpu:Z}=L;J.assertReach($.startPosition+$.tokens);let X=Z.device,Y=globalThis.__probeHooks,Q=$.timestamps?X.createQuerySet({type:"timestamp",count:$.tokens*2,label:"token_pass"}):null,H=R2(X,$);T2(L,$),await c1(X),Y.beginStep();let j=performance.now();for(let R=0;R<$.tokens;R+=1){let z=X.createCommandEncoder();J.encodeToken(z,1,Q===null?null:{querySet:Q,beginningOfPassWriteIndex:R*2,endOfPassWriteIndex:R*2+1});let T=(R+1)%$.readbackEvery===0,N=-1;if(T){N=(R+1)/$.readbackEvery-1;let y=await H.recycle(N);z.copyBufferToBuffer(J.buffers.sampled,($.startStep+R+1-$.readbackEvery)*4,y,0,$.readbackEvery*4)}if(X.queue.submit([z.finish()]),T)H.start(N)}await c1(X);let V=performance.now()-j,G=Y.endStep(),I=await H.drain();H.destroy();let D=await L8(X,Q,$.tokens,V);if(Q!==null)Q.destroy();return{wallMs:V,sampled:[...I],snapshot:G,...D}}async function $8(L,$){let{runtime:J,gpu:Z}=L,X=Z.device,Y=globalThis.__probeHooks;J.assertReach($.startPosition+$.tokens);let Q=Math.min(J.tokensPerPass,$.readbackEvery);if($.readbackEvery%Q)throw Error("Chunk does not divide readback cadence");let H=Math.ceil($.tokens/Q),j=$.timestamps?X.createQuerySet({type:"timestamp",count:H*2,label:"decode_chunk"}):null,V=R2(X,$);T2(L,$),await c1(X),Y.beginStep();let G=performance.now(),I=[];for(let E=0,t=0;E<$.tokens;E+=Q,t++){let e=Math.min(Q,$.tokens-E),k=(E+e)%$.readbackEvery===0,h=X.createCommandEncoder();J.encodeChunk(h,e,j?{querySet:j,beginningOfPassWriteIndex:t*2,endOfPassWriteIndex:t*2+1}:null),I.push(e);let L1=-1;if(k){L1=(E+e)/$.readbackEvery-1;let U1=await V.recycle(L1);h.copyBufferToBuffer(J.buffers.sampled,($.startStep+E+e-$.readbackEvery)*4,U1,0,$.readbackEvery*4)}if(X.queue.submit([h.finish()]),k)V.start(L1)}await c1(X);let D=performance.now()-G,R=Y.endStep(),z=await V.drain();V.destroy();let T=[],N=-1,y=-1;if(j){let E=await k0(X,j,H);N=0;for(let t=0;t<H;t++){let e=(E[t*2+1]-E[t*2])/1e6;T.push({tokens:I[t],ms:e}),N+=e}y=(E[H*2-1]-E[0])/1e6,j.destroy()}return{wallMs:D,snapshot:R,sampled:[...z],perTokenMs:[],gpuBusyMs:N,gpuSpanMs:y,gpuChunks:T}}var e6=e1.normEps;var J8=class{constructor(L){this.trie=this._build_trie(L)}_build_trie(L){let $=Object.create(null);for(let J of L){let Z=$;for(let X=0;X<J.length;++X){let Y=J[X];Z=Z[Y]??=Object.create(null)}Z.end=J}return $}split(L){let $=[],J=L.length,Z=0,X=0;while(X<J){let Y=this.trie,Q=null,H=X;while(H<J&&(Y=Y[L[H]])){if(Y.end)Q=Y.end;++H}if(Q){if(X>Z)$.push(L.slice(Z,X));$.push(Q),X+=Q.length,Z=X}else++X}if(Z<J)$.push(L.slice(Z));return $}},z2=J8,Z8=class{constructor(L){this.content=L.content,this.id=L.id,this.single_word=L.single_word??!1,this.lstrip=L.lstrip??!1,this.rstrip=L.rstrip??!1,this.special=L.special??!1,this.normalized=L.normalized??!this.special}},X8=Z8,w2=(()=>{let L=[...Array.from({length:94},(X,Y)=>Y+33),...Array.from({length:"¬".charCodeAt(0)-"¡".charCodeAt(0)+1},(X,Y)=>Y+"¡".charCodeAt(0)),...Array.from({length:"ÿ".charCodeAt(0)-"®".charCodeAt(0)+1},(X,Y)=>Y+"®".charCodeAt(0))],$=L.slice(),J=0;for(let X=0;X<256;++X)if(!L.includes(X))L.push(X),$.push(256+J),J+=1;let Z=$.map((X)=>String.fromCharCode(X));return Object.fromEntries(L.map((X,Y)=>[X,Z[Y]]))})(),Y8=(L)=>Object.fromEntries(Object.entries(L).map(([$,J])=>[J,$])),Q8=Y8(w2),E2=".,!?…。，、।۔،",H8=new Map([["(?i:'s|'t|'re|'ve|'m|'ll|'d)","(?:'([sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD]))"],["(?i:[sdmt]|ll|ve|re)","(?:[sS]|[dD]|[mM]|[tT]|[lL][lL]|[vV][eE]|[rR][eE])"],["[^\\r\\n\\p{L}\\p{N}]?+","[^\\r\\n\\p{L}\\p{N}]?"],["[^\\s\\p{L}\\p{N}]++","[^\\s\\p{L}\\p{N}]+"],["(?>\\p{Nd}{510})","(?:\\p{Nd}{510})"],["\\p{Nd}{3}+","(?:\\p{Nd}{3})+"],["\\G",""],[` ?[^(\\s|[${E2}])]+`,` ?[^\\s${E2}]+`]]),N0="\\p{P}\\u0021-\\u002F\\u003A-\\u0040\\u005B-\\u0060\\u007B-\\u007E",a0=(L)=>L.replace(/ \./g,".").replace(/ \?/g,"?").replace(/ \!/g,"!").replace(/ ,/g,",").replace(/ \' /g,"'").replace(/ n't/g,"n't").replace(/ 'm/g,"'m").replace(/ 's/g,"'s").replace(/ 've/g,"'ve").replace(/ 're/g,"'re"),T0=(L,$=!0)=>{if(L.Regex!==void 0){let J=L.Regex.replace(/\\([#&~])/g,"$1");J=J.replace(/\\A/g,"^").replace(/\\z/g,"$").replace(/\\Z/g,"(?=\\r?\\n?$)");for(let[Z,X]of H8)J=J.replaceAll(Z,X);try{return new RegExp(J,"gu")}catch(Z){if(!(Z instanceof SyntaxError)||!Z.message.toLowerCase().includes("invalid property name"))throw Z;let X=!1,Y=J.replace(/(\\[pP])\{([^}=]+)\}/g,(Q,H,j)=>{try{return new RegExp(`\\p{${j}}`,"u"),`${H}{${j}}`}catch{return X=!0,`${H}{Script=${j}}`}});if(!X)throw Z;try{return new RegExp(Y,"gu")}catch(Q){throw Z}}}else if(L.String!==void 0){let J=j8(L.String);return new RegExp($?J:`(${J})`,"gu")}else return console.warn("Unknown pattern type:",L),null},j8=(L)=>L.replace(/[.*+?^${}()|[\]\\]/g,"\\$&"),K8=(L,$,J)=>{let Z=[],X=0;while(X<L.length){if(Z.push(L[X]),($.get(L[X])??J)!==J){++X;continue}while(++X<L.length&&($.get(L[X])??J)===J)if($.get(Z.at(-1))!==J)Z[Z.length-1]+=L[X]}return Z},V8=(L)=>L>=19968&&L<=40959||L>=13312&&L<=19903||L>=131072&&L<=173791||L>=173824&&L<=177983||L>=177984&&L<=178207||L>=178208&&L<=183983||L>=63744&&L<=64255||L>=194560&&L<=195103,U8=(L)=>Number.isInteger(L)||typeof L==="bigint",W8=(L)=>{let $=0;for(let J of L)++$;return $},G8=(L)=>x2(L.toLowerCase()),M1=(...L)=>Array.prototype.concat.apply([],L),n0=(L)=>new Map(Object.entries(L)),q8=(L,$)=>{let J=[],Z=0;for(let X of L.matchAll($)){let Y=X[0];if(Z<X.index)J.push(L.slice(Z,X.index));if(Y.length>0)J.push(Y);Z=X.index+Y.length}if(Z<L.length)J.push(L.slice(Z));return J},x2=(L)=>L.replace(/\p{M}/gu,""),P2=(L,$,J=[])=>{if(!L||Array.isArray(L)||typeof L!=="object")return`${$} must be a valid object`;for(let Z of J)if(!(Z in L))return`${$} must contain a "${Z}" property`;return null},A8=(L)=>L.match(/\S+/g)||[],O8=class{constructor(){let L=function(...$){return L._call(...$)};return Object.setPrototypeOf(L,new.target.prototype)}},H0=O8,F8=class extends H0{constructor(L){super();this.config=L}_call(L){return this.normalize(L)}},y1=F8,C8=class extends y1{tokenize_chinese_chars(L){let $=[];for(let J=0;J<L.length;++J){let Z=L[J],X=Z.charCodeAt(0);if(V8(X))$.push(" "),$.push(Z),$.push(" ");else $.push(Z)}return $.join("")}strip_accents(L){return L.normalize("NFD").replace(/\p{Mn}/gu,"")}is_control(L){switch(L){case"\t":case`
`:case"\r":return!1;default:return/^\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}$/u.test(L)}}clean_text(L){let $=[];for(let J of L){let Z=J.charCodeAt(0);if(Z===0||Z===65533||this.is_control(J))continue;if(/^\s$/.test(J))$.push(" ");else $.push(J)}return $.join("")}normalize(L){if(this.config.clean_text)L=this.clean_text(L);if(this.config.handle_chinese_chars)L=this.tokenize_chinese_chars(L);if(this.config.lowercase){if(L=L.toLowerCase(),this.config.strip_accents!==!1)L=this.strip_accents(L)}else if(this.config.strip_accents)L=this.strip_accents(L);return L}},M8=C8,D8=class extends y1{constructor(L){super(L);this.charsmap=L.precompiled_charsmap??null}normalize(L){if(L=L.replace(/[\u0001-\u0008\u000B\u000E-\u001F\u007F\u008F\u009F]/gm,""),L=L.replace(/[\u0009\u000A\u000C\u000D\u00A0\u1680\u2000-\u200F\u2028\u2029\u202F\u205F\u2581\u3000\uFEFF\uFFFD]/gm," "),L.includes("～"))L=L.split("～").map((J)=>J.normalize("NFKC")).join("～");else L=L.normalize("NFKC");return L}},I8=D8,N8=class extends y1{constructor(L){super(L);this.normalizers=(L.normalizers??[]).map(($)=>_2($))}normalize(L){return this.normalizers.reduce(($,J)=>{return J?J.normalize($):$},L)}},T8=N8,R8=class extends y1{normalize(L){let $=T0(this.config.pattern??{});return $===null?L:L.replaceAll($,this.config.content??"")}},B8=R8,z8=class extends y1{constructor(){super(...arguments);this.form="NFC"}normalize(L){return L=L.normalize(this.form),L}},R0=z8,E8=class extends R0{constructor(){super(...arguments);this.form="NFC"}},P8=E8,v8=class extends R0{constructor(){super(...arguments);this.form="NFD"}},b8=v8,y8=class extends R0{constructor(){super(...arguments);this.form="NFKC"}},S8=y8,w8=class extends R0{constructor(){super(...arguments);this.form="NFKD"}},x8=w8,_8=class extends y1{normalize(L){if(this.config.strip_left&&this.config.strip_right)L=L.trim();else{if(this.config.strip_left)L=L.trimStart();if(this.config.strip_right)L=L.trimEnd()}return L}},m8=_8,f8=class extends y1{normalize(L){return x2(L)}},k8=f8,g8=class extends y1{normalize(L){return L.toLowerCase()}},h8=g8,u8=class extends y1{normalize(L){return L=this.config.prepend+L,L}},l8=u8;function p8(L){if(L===null)return null;switch(L.type){case"BertNormalizer":return new M8(L);case"Precompiled":return new I8(L);case"Sequence":return new T8(L);case"Replace":return new B8(L);case"NFC":return new P8(L);case"NFD":return new b8(L);case"NFKC":return new S8(L);case"NFKD":return new x8(L);case"Strip":return new m8(L);case"StripAccents":return new k8(L);case"Lowercase":return new h8(L);case"Prepend":return new l8(L);default:throw Error(`Unknown Normalizer type: ${L.type}`)}}var _2=p8,d8=class extends H0{pre_tokenize(L,$){return(Array.isArray(L)?L.map((J)=>this.pre_tokenize_text(J,$)):this.pre_tokenize_text(L,$)).flat()}_call(L,$){return this.pre_tokenize(L,$)}},D1=d8,c8=class extends D1{constructor(L){super();this.config=L,this.add_prefix_space=this.config.add_prefix_space??!1,this.trim_offsets=this.config.trim_offsets??!1,this.use_regex=this.config.use_regex??!0,this.pattern=/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu,this.byte_encoder=w2,this.text_encoder=new TextEncoder}pre_tokenize_text(L,$){if(this.add_prefix_space&&!L.startsWith(" "))L=" "+L;return(this.use_regex?L.match(this.pattern)||[]:[L]).map((Z)=>Array.from(this.text_encoder.encode(Z),(X)=>this.byte_encoder[X]).join(""))}},a8=c8,n8=class extends D1{pre_tokenize_text(L,$){return L.match(/\w+|[^\w\s]+/g)||[]}},r8=n8,i8=class extends D1{constructor(L){super();this.replacement=L.replacement??"▁",this.str_rep=L.str_rep||this.replacement,this.prepend_scheme=L.prepend_scheme??"always"}pre_tokenize_text(L,$){let{section_index:J=void 0}=$??{},Z=L.replaceAll(" ",this.str_rep);if(!Z.startsWith(this.replacement)&&(this.prepend_scheme==="always"||this.prepend_scheme==="first"&&J===0))Z=this.str_rep+Z;return[Z]}},t8=i8,s8=class extends D1{constructor(L){super();this.config=L,this.pattern=T0(this.config.pattern??{},this.config.invert??!0)}pre_tokenize_text(L){if(this.pattern===null)return[];if(this.config.invert)return L.match(this.pattern)||[];else if(this.config.behavior?.toLowerCase()==="removed")return L.split(this.pattern).filter(($)=>$);else return q8(L,this.pattern)}},o8=s8,e8=class extends D1{constructor(L){super();this.config=L,this.pattern=new RegExp(`[^${N0}]+|[${N0}]+`,"gu")}pre_tokenize_text(L){return L.match(this.pattern)||[]}},L5=e8,$5=class extends D1{constructor(L){super();this.config=L;let $=`[^\\d]+|\\d${this.config.individual_digits?"":"+"}`;this.pattern=new RegExp($,"gu")}pre_tokenize_text(L){return L.match(this.pattern)||[]}},J5=$5,Z5=class extends D1{constructor(){super();this.pattern=new RegExp(`[^\\s${N0}]+|[${N0}]`,"gu")}pre_tokenize_text(L,$){return L.trim().match(this.pattern)||[]}},X5=Z5,Y5=class extends D1{constructor(L){super();this.config=L,this.pattern=T0(this.config.pattern??{}),this.content=this.config.content??""}pre_tokenize_text(L){if(this.pattern===null)return[L];return[L.replaceAll(this.pattern,this.config.content??"")]}},Q5=Y5,H5=class extends D1{constructor(L){super();this.tokenizers=(L.pretokenizers??[]).map(($)=>m2($))}pre_tokenize_text(L,$){return this.tokenizers.reduce((J,Z)=>{return Z?Z.pre_tokenize(J,$):J},[L])}},j5=H5,K5=class extends D1{pre_tokenize_text(L){return A8(L)}},V5=K5,U5=class extends D1{constructor(L){super();this.config=L,this._length=L.length}pre_tokenize_text(L){let $=[];for(let J=0;J<L.length;J+=this._length)$.push(L.slice(J,J+this._length));return $}},W5=U5;function G5(L){if(L===null)return null;switch(L.type){case"BertPreTokenizer":return new X5;case"Sequence":return new j5(L);case"Whitespace":return new r8;case"WhitespaceSplit":return new V5;case"Metaspace":return new t8(L);case"ByteLevel":return new a8(L);case"Split":return new o8(L);case"Punctuation":return new L5(L);case"Digits":return new J5(L);case"Replace":return new Q5(L);case"FixedLength":return new W5(L);default:throw Error(`Unknown PreTokenizer type: ${L.type}`)}}var m2=G5,q5=class extends H0{constructor(L){super();this.config=L,this.vocab=[],this.tokens_to_ids=new Map,this.unk_token_id=void 0,this.unk_token=void 0,this.end_of_word_suffix=void 0,this.fuse_unk=this.config.fuse_unk??!1}_call(L){let $=this.encode(L);if(this.fuse_unk)$=K8($,this.tokens_to_ids,this.unk_token_id);return $}},B0=q5,A5=class extends B0{constructor(L){super(L);this.max_input_chars_per_word=100,this.tokens_to_ids=n0(L.vocab),this.unk_token_id=this.tokens_to_ids.get(L.unk_token),this.unk_token=L.unk_token,this.max_input_chars_per_word=L.max_input_chars_per_word??100,this.vocab=Array(this.tokens_to_ids.size);for(let[$,J]of this.tokens_to_ids)this.vocab[J]=$}encode(L){let $=[];for(let J of L){let Z=[...J];if(Z.length>this.max_input_chars_per_word){$.push(this.unk_token);continue}let X=!1,Y=0,Q=[];while(Y<Z.length){let H=Z.length,j=null;while(Y<H){let V=Z.slice(Y,H).join("");if(Y>0)V=this.config.continuing_subword_prefix+V;if(this.tokens_to_ids.has(V)){j=V;break}--H}if(j===null){X=!0;break}Q.push(j),Y=H}if(X)$.push(this.unk_token);else $.push(...Q)}return $}},v2=A5,b2=class L{constructor($,J){this.is_leaf=$,this.children=J}static default(){return new L(!1,new Map)}},O5=class{constructor(){this.root=b2.default()}extend(L){for(let $ of L)this.push($)}push(L){let $=this.root;for(let J of L){let Z=$.children.get(J);if(Z===void 0)Z=b2.default(),$.children.set(J,Z);$=Z}$.is_leaf=!0}*common_prefix_search(L){let $=this.root;if($===void 0)return;let J="";for(let Z of L){if(J+=Z,$=$.children.get(Z),$===void 0)return;if($.is_leaf)yield J}}},F5=O5,c0=class L{constructor($,J,Z,X,Y){this.token_id=$,this.node_id=J,this.pos=Z,this.length=X,this.score=Y,this.prev=null,this.backtrace_score=0}clone(){let $=new L(this.token_id,this.node_id,this.pos,this.length,this.score);return $.prev=this.prev,$.backtrace_score=this.backtrace_score,$}},C5=class{constructor(L,$,J){this.chars=Array.from(L),this.len=this.chars.length,this.bos_token_id=$,this.eos_token_id=J,this.nodes=[],this.begin_nodes=Array.from({length:this.len+1},()=>[]),this.end_nodes=Array.from({length:this.len+1},()=>[]);let Z=new c0(this.bos_token_id??0,0,0,0,0),X=new c0(this.eos_token_id??0,1,this.len,0,0);this.nodes.push(Z.clone()),this.nodes.push(X.clone()),this.begin_nodes[this.len].push(X),this.end_nodes[0].push(Z)}insert(L,$,J,Z){let X=this.nodes.length,Y=new c0(Z,X,L,$,J);this.begin_nodes[L].push(Y),this.end_nodes[L+$].push(Y),this.nodes.push(Y)}viterbi(){let L=this.len,$=0;while($<=L){if(this.begin_nodes[$].length==0)return[];for(let Q of this.begin_nodes[$]){Q.prev=null;let H=0,j=null;for(let V of this.end_nodes[$]){let G=V.backtrace_score+Q.score;if(j===null||G>H)j=V.clone(),H=G}if(j!==null)Q.prev=j,Q.backtrace_score=H;else return[]}++$}let J=[],X=this.begin_nodes[L][0].prev;if(X===null)return[];let Y=X.clone();while(Y.prev!==null)J.push(Y.clone()),Y=Y.clone().prev.clone();return J.reverse(),J}piece(L){return this.chars.slice(L.pos,L.pos+L.length).join("")}tokens(){return this.viterbi().map(($)=>this.piece($))}token_ids(){return this.viterbi().map(($)=>$.token_id)}},M5=C5;function D5(L){if(L.length===0)throw Error("Array must not be empty");let $=L[0],J=0;for(let Z=1;Z<L.length;++Z)if(L[Z]<$)$=L[Z],J=Z;return[$,J]}var I5=class extends B0{constructor(L,$){super(L);let J=L.vocab.length;this.vocab=Array(J),this.scores=Array(J);for(let Z=0;Z<J;++Z)[this.vocab[Z],this.scores[Z]]=L.vocab[Z];this.unk_token_id=L.unk_id,this.unk_token=this.vocab[L.unk_id],this.tokens_to_ids=new Map(this.vocab.map((Z,X)=>[Z,X])),this.bos_token=" ",this.bos_token_id=this.tokens_to_ids.get(this.bos_token),this.eos_token=$,this.eos_token_id=this.tokens_to_ids.get(this.eos_token),this.unk_token=this.vocab[this.unk_token_id],this.min_score=D5(this.scores)[0],this.unk_score=this.min_score-10,this.scores[this.unk_token_id]=this.unk_score,this.trie=new F5,this.trie.extend(this.vocab),this.fuse_unk=!0}populate_nodes(L){let $=L.chars,J=1,Z=0;while(Z<$.length){let X=!1,Y=[],Q=$.slice(Z).join(""),H=this.trie.common_prefix_search(Q);for(let j of H){Y.push(j);let V=this.tokens_to_ids.get(j),G=this.scores[V],I=W8(j);if(L.insert(Z,I,G,V),!X&&I===1)X=!0}if(!X)L.insert(Z,1,this.unk_score,this.unk_token_id);Z+=1}}tokenize(L){let $=new M5(L,this.bos_token_id,this.eos_token_id);return this.populate_nodes($),$.tokens()}encode(L){let $=[];for(let J of L){let Z=this.tokenize(J);$.push(...Z)}return $}},y2=I5,N5=class{constructor(L=(J,Z)=>J>Z,$=1/0){this._heap=[],this._comparator=L,this._max_size=$}get size(){return this._heap.length}is_empty(){return this.size===0}peek(){return this._heap[0]}push(...L){return this.extend(L)}extend(L){for(let $ of L)if(this.size<this._max_size)this._heap.push($),this._sift_up();else{let J=this._smallest();if(this._comparator($,this._heap[J]))this._heap[J]=$,this._sift_up_from(J)}return this.size}pop(){let L=this.peek(),$=this.size-1;if($>0)this._swap(0,$);return this._heap.pop(),this._sift_down(),L}replace(L){let $=this.peek();return this._heap[0]=L,this._sift_down(),$}_parent(L){return(L+1>>>1)-1}_left(L){return(L<<1)+1}_right(L){return L+1<<1}_greater(L,$){return this._comparator(this._heap[L],this._heap[$])}_swap(L,$){let J=this._heap[L];this._heap[L]=this._heap[$],this._heap[$]=J}_sift_up(){this._sift_up_from(this.size-1)}_sift_up_from(L){while(L>0&&this._greater(L,this._parent(L)))this._swap(L,this._parent(L)),L=this._parent(L)}_sift_down(){let L=0;while(this._left(L)<this.size&&this._greater(this._left(L),L)||this._right(L)<this.size&&this._greater(this._right(L),L)){let $=this._right(L)<this.size&&this._greater(this._right(L),this._left(L))?this._right(L):this._left(L);this._swap(L,$),L=$}}_smallest(){return 2**Math.floor(Math.log2(this.size))-1}},T5=N5,R5=class{constructor(L){this.capacity=L,this.cache=new Map}get(L){if(!this.cache.has(L))return;let $=this.cache.get(L);return this.cache.delete(L),this.cache.set(L,$),$}put(L,$){if(this.cache.has(L))this.cache.delete(L);if(this.cache.set(L,$),this.cache.size>this.capacity)this.cache.delete(this.cache.keys().next().value)}clear(){this.cache.clear()}},B5=R5,z5=class extends B0{constructor(L){super(L);this.tokens_to_ids=n0(L.vocab),this.unk_token_id=this.tokens_to_ids.get(L.unk_token),this.unk_token=L.unk_token,this.vocab=Array(this.tokens_to_ids.size);for(let[J,Z]of this.tokens_to_ids)this.vocab[Z]=J;let $=Array.isArray(L.merges[0]);if(this.merges=$?L.merges:L.merges.map((J)=>J.split(" ",2)),this.bpe_ranks=new Map(this.merges.map((J,Z)=>[JSON.stringify(J),Z])),this.end_of_word_suffix=L.end_of_word_suffix,this.continuing_subword_suffix=L.continuing_subword_suffix??null,this.byte_fallback=this.config.byte_fallback??!1,this.byte_fallback)this.text_encoder=new TextEncoder;this.ignore_merges=this.config.ignore_merges??!1,this.max_length_to_cache=256,this.cache_capacity=1e4,this.cache=new B5(this.cache_capacity)}clear_cache(){this.cache.clear()}bpe(L){if(L.length===0)return[];let $=this.cache.get(L);if($!==void 0)return $;let J=Array.from(L);if(this.end_of_word_suffix)J[J.length-1]+=this.end_of_word_suffix;let Z=[];if(J.length>1){let X=new T5((H,j)=>H.score<j.score),Y={token:J[0],bias:0,prev:null,next:null},Q=Y;for(let H=1;H<J.length;++H){let j={bias:H/J.length,token:J[H],prev:Q,next:null};Q.next=j,this.add_node(X,Q),Q=j}while(!X.is_empty()){let H=X.pop();if(H.deleted||!H.next||H.next.deleted)continue;if(H.deleted=!0,H.next.deleted=!0,H.prev){let V={...H.prev};if(H.prev.deleted=!0,H.prev=V,V.prev)V.prev.next=V;else Y=V}let j={token:H.token+H.next.token,bias:H.bias,prev:H.prev,next:H.next.next};if(j.prev)j.prev.next=j,this.add_node(X,j.prev);else Y=j;if(j.next)j.next.prev=j,this.add_node(X,j)}for(let H=Y;H!==null;H=H.next)Z.push(H.token)}else Z=J;if(this.continuing_subword_suffix)for(let X=0;X<Z.length-1;++X)Z[X]+=this.continuing_subword_suffix;if(L.length<this.max_length_to_cache)this.cache.put(L,Z);return Z}add_node(L,$){let J=this.bpe_ranks.get(JSON.stringify([$.token,$.next.token]));if(J!==void 0)$.score=J+$.bias,L.push($)}encode(L){let $=[];for(let J of L){if(this.ignore_merges&&this.tokens_to_ids.has(J)){$.push(J);continue}let Z=this.bpe(J);for(let X of Z)if(this.tokens_to_ids.has(X))$.push(X);else if(this.byte_fallback){let Y=Array.from(this.text_encoder.encode(X)).map((Q)=>`<0x${Q.toString(16).toUpperCase().padStart(2,"0")}>`);if(Y.every((Q)=>this.tokens_to_ids.has(Q)))$.push(...Y);else if(this.unk_token!=null)$.push(this.unk_token)}else if(this.unk_token!=null)$.push(this.unk_token)}return $}},S2=z5,E5=class extends B0{constructor(L,$){super(L);let J=L.vocab;this.tokens_to_ids=n0($.target_lang?J[$.target_lang]:J),this.bos_token=$.bos_token,this.bos_token_id=this.tokens_to_ids.get(this.bos_token),this.eos_token=$.eos_token,this.eos_token_id=this.tokens_to_ids.get(this.eos_token),this.pad_token=$.pad_token,this.pad_token_id=this.tokens_to_ids.get(this.pad_token),this.unk_token=$.unk_token,this.unk_token_id=this.tokens_to_ids.get(this.unk_token),this.vocab=Array(this.tokens_to_ids.size);for(let[Z,X]of this.tokens_to_ids)this.vocab[X]=Z}encode(L){return L}},P5=E5;function v5(L,$){switch(L.type){case"WordPiece":return new v2(L);case"Unigram":return new y2(L,$.eos_token);case"BPE":return new S2(L);default:if(L.vocab)if(Array.isArray(L.vocab))return new y2(L,$.eos_token);else if(Object.hasOwn(L,"continuing_subword_prefix")&&Object.hasOwn(L,"unk_token"))if(Object.hasOwn(L,"merges"))return new S2(L);else return new v2(L);else return new P5(L,{target_lang:$.target_lang,bos_token:$.bos_token,eos_token:$.eos_token,pad_token:$.pad_token,unk_token:$.unk_token});throw Error(`Unknown TokenizerModel type: ${L?.type}`)}}var b5=v5,y5=class extends H0{constructor(L){super();this.config=L}_call(L,...$){return this.post_process(L,...$)}},j0=y5,S5=class extends j0{post_process(L,$=null,J=!0){let Z=$===null?this.config.single:this.config.pair,X=[],Y=[];for(let Q of Z)if("SpecialToken"in Q){if(J)X.push(Q.SpecialToken.id),Y.push(Q.SpecialToken.type_id)}else if("Sequence"in Q){if(Q.Sequence.id==="A")X=M1(X,L),Y=M1(Y,Array(L.length).fill(Q.Sequence.type_id));else if(Q.Sequence.id==="B")X=M1(X,$),Y=M1(Y,Array($.length).fill(Q.Sequence.type_id))}return{tokens:X,token_type_ids:Y}}},w5=S5,x5=class extends j0{post_process(L,$=null){return{tokens:L,tokens_pair:$}}},_5=x5,m5=class extends j0{constructor(L){super(L);this.sep=L.sep,this.cls=L.cls}post_process(L,$=null,J=!0){if(J)L=M1([this.cls[0]],L,[this.sep[0]]);let Z=Array(L.length).fill(0);if($){let X=[],Y=J?[this.sep[0]]:[];L=M1(L,X,$,Y),Z=M1(Z,Array($.length+X.length+Y.length).fill(1))}return{tokens:L,token_type_ids:Z}}},f5=m5,k5=class extends j0{constructor(L){super(L);this.sep=L.sep,this.cls=L.cls}post_process(L,$,J=!0){if(J)L=M1([this.cls[0]],L,[this.sep[0]]);let Z=Array(L.length).fill(0);if($){let X=J?[this.sep[0]]:[],Y=J?[this.sep[0]]:[];L=M1(L,X,$,Y),Z=M1(Z,Array($.length+X.length+Y.length).fill(1))}return{tokens:L,token_type_ids:Z}}},g5=k5,h5=class extends j0{constructor(L){super(L);this.processors=(L.processors??[]).map(($)=>f2($))}post_process(L,$=null,J=!0){let Z={tokens:L,tokens_pair:$};for(let X of this.processors)Z=X.post_process(Z.tokens,Z.tokens_pair,J);return Z}},u5=h5;function l5(L){if(L===null)return null;switch(L.type){case"TemplateProcessing":return new w5(L);case"ByteLevel":return new _5(L);case"BertProcessing":return new f5(L);case"RobertaProcessing":return new g5(L);case"Sequence":return new u5(L);default:throw Error(`Unknown PostProcessor type: ${L.type}`)}}var f2=l5,p5=class extends H0{constructor(L){super();this.config=L,this.added_tokens=[],this.end_of_word_suffix=null,this.trim_offsets="trim_offsets"in L?L.trim_offsets:!1}_call(L){return this.decode(L)}decode(L){return this.decode_chain(L).join("")}},T1=p5,d5=class extends T1{constructor(L){super(L);this.byte_decoder=Q8,this.text_decoder=new TextDecoder("utf-8",{fatal:!1,ignoreBOM:!0}),this.end_of_word_suffix=null}convert_tokens_to_string(L){let $=L.join(""),J=new Uint8Array([...$].map((Z)=>this.byte_decoder[Z]));return this.text_decoder.decode(J)}decode_chain(L){let $=[],J=[];for(let Z of L)if(this.added_tokens.find((X)=>X.content===Z)!==void 0){if(J.length>0)$.push(this.convert_tokens_to_string(J)),J=[];$.push(Z)}else J.push(Z);if(J.length>0)$.push(this.convert_tokens_to_string(J));return $}},c5=d5,a5=class extends T1{constructor(L){super(L);this.cleanup=L.cleanup}decode_chain(L){return L.map(($,J)=>{if(J!==0){let Z=this.config.prefix;if(Z&&$.startsWith(Z))$=$.replace(Z,"");else $=" "+$}if(this.cleanup)$=a0($);return $})}},n5=a5,r5=class extends T1{constructor(L){super(L);this.replacement=L.replacement??"▁"}decode_chain(L){let $=[];for(let J=0;J<L.length;++J){let Z=L[J].replaceAll(this.replacement," ");if(J==0&&Z.startsWith(" "))Z=Z.substring(1);$.push(Z)}return $}},i5=r5,t5=class extends T1{constructor(L){super(L);this.suffix=L.suffix??""}decode_chain(L){return L.map(($,J)=>{return $.replaceAll(this.suffix,J===L.length-1?"":" ")})}},s5=t5,o5=class extends T1{constructor(L){super(L);this.pad_token=L.pad_token??"",this.word_delimiter_token=L.word_delimiter_token??"",this.cleanup=L.cleanup}convert_tokens_to_string(L){if(L.length===0)return"";let $=[L[0]];for(let X=1;X<L.length;++X)if(L[X]!==$.at(-1))$.push(L[X]);let Z=$.filter((X)=>X!==this.pad_token).join("");if(this.cleanup)Z=a0(Z).replaceAll(this.word_delimiter_token," ").trim();return Z}decode_chain(L){return[this.convert_tokens_to_string(L)]}},e5=o5,L4=class extends T1{constructor(L){super(L);this.decoders=(L.decoders??[]).map(($)=>k2($))}decode_chain(L){return this.decoders.reduce(($,J)=>{return J.decode_chain($)},L)}},$4=L4,J4=class extends T1{decode_chain(L){let $=T0(this.config.pattern),J=this.config.content??"";return $===null?L:L.map((Z)=>Z.replaceAll($,J))}},Z4=J4,X4=class extends T1{decode_chain(L){return[L.join("")]}},Y4=X4,Q4=class extends T1{constructor(L){super(L);this.content=L.content??"",this.start=L.start??0,this.stop=L.stop??0}decode_chain(L){return L.map(($)=>{let J=0;for(let X=0;X<this.start;++X)if($[X]===this.content){J=X+1;continue}else break;let Z=$.length;for(let X=0;X<this.stop;++X){let Y=$.length-X-1;if($[Y]===this.content){Z=Y;continue}else break}return $.slice(J,Z)})}},H4=Q4,j4=class extends T1{constructor(L){super(L);this.text_decoder=new TextDecoder}decode_chain(L){let $=[],J=[];for(let Z of L){let X=null;if(Z.length===6&&Z.startsWith("<0x")&&Z.endsWith(">")){let Y=parseInt(Z.slice(3,5),16);if(!isNaN(Y))X=Y}if(X!==null)J.push(X);else{if(J.length>0){let Y=this.text_decoder.decode(Uint8Array.from(J));$.push(Y),J=[]}$.push(Z)}}if(J.length>0){let Z=this.text_decoder.decode(Uint8Array.from(J));$.push(Z),J=[]}return $}},K4=j4;function V4(L){if(L===null)return null;switch(L.type){case"ByteLevel":return new c5(L);case"WordPiece":return new n5(L);case"Metaspace":return new i5(L);case"BPEDecoder":return new s5(L);case"CTC":return new e5(L);case"Sequence":return new $4(L);case"Replace":return new Z4(L);case"Fuse":return new Y4(L);case"Strip":return new H4(L);case"ByteFallback":return new K4(L);default:throw Error(`Unknown Decoder type: ${L.type}`)}}var k2=V4,U4=class{constructor(L,$){let J=P2(L,"Tokenizer",["model","decoder","post_processor","pre_tokenizer","normalizer"]);if(J)throw Error(J);let Z=P2($,"Config");if(Z)throw Error(Z);this.tokenizer=L,this.config=$,this.normalizer=_2(this.tokenizer.normalizer),this.pre_tokenizer=m2(this.tokenizer.pre_tokenizer),this.model=b5(this.tokenizer.model,this.config),this.post_processor=f2(this.tokenizer.post_processor),this.decoder=k2(this.tokenizer.decoder),this.special_tokens=[],this.all_special_ids=[],this.added_tokens=[];let X=[],Y=[];this.added_tokens_map=new Map;for(let Q of this.tokenizer.added_tokens){let H=new X8(Q);if(this.added_tokens.push(H),this.model.tokens_to_ids.set(H.content,H.id),this.model.vocab[H.id]=H.content,H.special)this.special_tokens.push(H.content),this.all_special_ids.push(H.id);if(this.added_tokens_map.set(H.content,H),H.normalized&&this.normalizer!==null){let j=this.normalizer(H.content);Y.push(j),this.added_tokens_map.set(j,H)}else X.push(H.content)}if((this.config.additional_special_tokens??[]).forEach((Q)=>{if(!this.special_tokens.includes(Q))this.special_tokens.push(Q)}),this.decoder)this.decoder.added_tokens=this.added_tokens,this.decoder.end_of_word_suffix=this.model.end_of_word_suffix;this.splitter_unnormalized=new z2(X),this.splitter_normalized=new z2(Y),this.remove_space=this.config.remove_space,this.clean_up_tokenization_spaces=this.config.clean_up_tokenization_spaces??!0,this.do_lowercase_and_remove_accent=this.config.do_lowercase_and_remove_accent??!1}encode(L,{text_pair:$=null,add_special_tokens:J=!0,return_token_type_ids:Z=null}={}){let{tokens:X,token_type_ids:Y}=this.tokenize_helper(L,{text_pair:$,add_special_tokens:J}),Q=X.map((j)=>this.added_tokens_map.get(j)?.id??this.model.tokens_to_ids.get(j)??this.model.unk_token_id),H={ids:Q,tokens:X,attention_mask:Array(Q.length).fill(1)};if(Z&&Y)H.token_type_ids=Y;return H}decode(L,$={}){if(!Array.isArray(L)||L.length===0||!U8(L[0]))throw Error("token_ids must be a non-empty array of integers.");let J=L.map((X)=>this.model.vocab[Number(X)]??this.model.unk_token);if($.skip_special_tokens)J=J.filter((X)=>!this.special_tokens.includes(X));let Z=this.decoder?this.decoder(J):J.join(" ");if(this.decoder&&this.decoder.end_of_word_suffix){if(Z=Z.replaceAll(this.decoder.end_of_word_suffix," "),$.skip_special_tokens)Z=Z.trim()}if($.clean_up_tokenization_spaces??this.clean_up_tokenization_spaces)Z=a0(Z);return Z}tokenize(L,{text_pair:$=null,add_special_tokens:J=!1}={}){return this.tokenize_helper(L,{text_pair:$,add_special_tokens:J}).tokens}encode_text(L){if(L===null)return null;let $=this.splitter_unnormalized.split(L);return $.forEach((J,Z)=>{let X=this.added_tokens_map.get(J);if(X){if(X.lstrip&&Z>0)$[Z-1]=$[Z-1].trimEnd();if(X.rstrip&&Z<$.length-1)$[Z+1]=$[Z+1].trimStart()}}),$.flatMap((J,Z)=>{if(J.length===0)return[];if(this.added_tokens_map.has(J))return[J];if(this.remove_space===!0)J=J.trim().split(/\s+/).join(" ");if(this.do_lowercase_and_remove_accent)J=G8(J);if(this.normalizer!==null)J=this.normalizer(J);if(J.length===0)return[];let X=this.splitter_normalized.split(J);return X.forEach((Y,Q)=>{let H=this.added_tokens_map.get(Y);if(H){if(H.lstrip&&Q>0)X[Q-1]=X[Q-1].trimEnd();if(H.rstrip&&Q<X.length-1)X[Q+1]=X[Q+1].trimStart()}}),X.flatMap((Y)=>{if(Y.length===0)return[];if(this.added_tokens_map.has(Y))return[Y];let Q=this.pre_tokenizer!==null?this.pre_tokenizer(Y,{section_index:Z}):[Y];return this.model(Q)})})}tokenize_helper(L,{text_pair:$=null,add_special_tokens:J=!0}){let Z=this.encode_text(L),X=this.encode_text($||null);return this.post_processor?this.post_processor(Z,X,J):{tokens:M1(Z??[],X??[])}}token_to_id(L){return this.model.tokens_to_ids.get(L)}id_to_token(L){return this.model.vocab[L]}get_added_tokens_decoder(){let L=new Map;for(let $ of this.added_tokens)L.set($.id,$);return L}get_vocab(L=!0){let $=new Map;for(let J=0;J<this.model.vocab.length;++J){let Z=this.model.vocab[J];if(L||!this.added_tokens_map.has(Z))$.set(Z,J)}return $}},g2=U4;var h1={runtime:{shape:{layerTypes:["conv","conv","attention","conv","attention","conv","attention","conv","attention","conv","attention","conv","attention","conv"],hidden:1024,ffn:2560,vocab:65536,heads:16,kvHeads:8,headDim:64,convCache:3,quantBits:4,quantBlock:32},weightsUrl:"/weights.bin",weightBytes:168647680,sections:[{name:"L0.conv.in_proj.quant",offset:0,bytes:1572864},{name:"L0.conv.in_proj.scales",offset:1572864,bytes:393216},{name:"L0.conv.out_proj.quant",offset:1966080,bytes:524288},{name:"L0.conv.out_proj.scales",offset:2490368,bytes:131072},{name:"L0.mlp.gate_up.quant",offset:2621440,bytes:2621440},{name:"L0.mlp.gate_up.scales",offset:5242880,bytes:655360},{name:"L0.mlp.down.quant",offset:5898240,bytes:1310720},{name:"L0.mlp.down.scales",offset:7208960,bytes:327680},{name:"L1.conv.in_proj.quant",offset:7536640,bytes:1572864},{name:"L1.conv.in_proj.scales",offset:9109504,bytes:393216},{name:"L1.conv.out_proj.quant",offset:9502720,bytes:524288},{name:"L1.conv.out_proj.scales",offset:10027008,bytes:131072},{name:"L1.mlp.gate_up.quant",offset:10158080,bytes:2621440},{name:"L1.mlp.gate_up.scales",offset:12779520,bytes:655360},{name:"L1.mlp.down.quant",offset:13434880,bytes:1310720},{name:"L1.mlp.down.scales",offset:14745600,bytes:327680},{name:"L2.attn.qkv.quant",offset:15073280,bytes:1048576},{name:"L2.attn.qkv.scales",offset:16121856,bytes:262144},{name:"L2.attn.o_proj.quant",offset:16384000,bytes:524288},{name:"L2.attn.o_proj.scales",offset:16908288,bytes:131072},{name:"L2.mlp.gate_up.quant",offset:17039360,bytes:2621440},{name:"L2.mlp.gate_up.scales",offset:19660800,bytes:655360},{name:"L2.mlp.down.quant",offset:20316160,bytes:1310720},{name:"L2.mlp.down.scales",offset:21626880,bytes:327680},{name:"L3.conv.in_proj.quant",offset:21954560,bytes:1572864},{name:"L3.conv.in_proj.scales",offset:23527424,bytes:393216},{name:"L3.conv.out_proj.quant",offset:23920640,bytes:524288},{name:"L3.conv.out_proj.scales",offset:24444928,bytes:131072},{name:"L3.mlp.gate_up.quant",offset:24576000,bytes:2621440},{name:"L3.mlp.gate_up.scales",offset:27197440,bytes:655360},{name:"L3.mlp.down.quant",offset:27852800,bytes:1310720},{name:"L3.mlp.down.scales",offset:29163520,bytes:327680},{name:"L4.attn.qkv.quant",offset:29491200,bytes:1048576},{name:"L4.attn.qkv.scales",offset:30539776,bytes:262144},{name:"L4.attn.o_proj.quant",offset:30801920,bytes:524288},{name:"L4.attn.o_proj.scales",offset:31326208,bytes:131072},{name:"L4.mlp.gate_up.quant",offset:31457280,bytes:2621440},{name:"L4.mlp.gate_up.scales",offset:34078720,bytes:655360},{name:"L4.mlp.down.quant",offset:34734080,bytes:1310720},{name:"L4.mlp.down.scales",offset:36044800,bytes:327680},{name:"L5.conv.in_proj.quant",offset:36372480,bytes:1572864},{name:"L5.conv.in_proj.scales",offset:37945344,bytes:393216},{name:"L5.conv.out_proj.quant",offset:38338560,bytes:524288},{name:"L5.conv.out_proj.scales",offset:38862848,bytes:131072},{name:"L5.mlp.gate_up.quant",offset:38993920,bytes:2621440},{name:"L5.mlp.gate_up.scales",offset:41615360,bytes:655360},{name:"L5.mlp.down.quant",offset:42270720,bytes:1310720},{name:"L5.mlp.down.scales",offset:43581440,bytes:327680},{name:"L6.attn.qkv.quant",offset:43909120,bytes:1048576},{name:"L6.attn.qkv.scales",offset:44957696,bytes:262144},{name:"L6.attn.o_proj.quant",offset:45219840,bytes:524288},{name:"L6.attn.o_proj.scales",offset:45744128,bytes:131072},{name:"L6.mlp.gate_up.quant",offset:45875200,bytes:2621440},{name:"L6.mlp.gate_up.scales",offset:48496640,bytes:655360},{name:"L6.mlp.down.quant",offset:49152000,bytes:1310720},{name:"L6.mlp.down.scales",offset:50462720,bytes:327680},{name:"L7.conv.in_proj.quant",offset:50790400,bytes:1572864},{name:"L7.conv.in_proj.scales",offset:52363264,bytes:393216},{name:"L7.conv.out_proj.quant",offset:52756480,bytes:524288},{name:"L7.conv.out_proj.scales",offset:53280768,bytes:131072},{name:"L7.mlp.gate_up.quant",offset:53411840,bytes:2621440},{name:"L7.mlp.gate_up.scales",offset:56033280,bytes:655360},{name:"L7.mlp.down.quant",offset:56688640,bytes:1310720},{name:"L7.mlp.down.scales",offset:57999360,bytes:327680},{name:"L8.attn.qkv.quant",offset:58327040,bytes:1048576},{name:"L8.attn.qkv.scales",offset:59375616,bytes:262144},{name:"L8.attn.o_proj.quant",offset:59637760,bytes:524288},{name:"L8.attn.o_proj.scales",offset:60162048,bytes:131072},{name:"L8.mlp.gate_up.quant",offset:60293120,bytes:2621440},{name:"L8.mlp.gate_up.scales",offset:62914560,bytes:655360},{name:"L8.mlp.down.quant",offset:63569920,bytes:1310720},{name:"L8.mlp.down.scales",offset:64880640,bytes:327680},{name:"L9.conv.in_proj.quant",offset:65208320,bytes:1572864},{name:"L9.conv.in_proj.scales",offset:66781184,bytes:393216},{name:"L9.conv.out_proj.quant",offset:67174400,bytes:524288},{name:"L9.conv.out_proj.scales",offset:67698688,bytes:131072},{name:"L9.mlp.gate_up.quant",offset:67829760,bytes:2621440},{name:"L9.mlp.gate_up.scales",offset:70451200,bytes:655360},{name:"L9.mlp.down.quant",offset:71106560,bytes:1310720},{name:"L9.mlp.down.scales",offset:72417280,bytes:327680},{name:"L10.attn.qkv.quant",offset:72744960,bytes:1048576},{name:"L10.attn.qkv.scales",offset:73793536,bytes:262144},{name:"L10.attn.o_proj.quant",offset:74055680,bytes:524288},{name:"L10.attn.o_proj.scales",offset:74579968,bytes:131072},{name:"L10.mlp.gate_up.quant",offset:74711040,bytes:2621440},{name:"L10.mlp.gate_up.scales",offset:77332480,bytes:655360},{name:"L10.mlp.down.quant",offset:77987840,bytes:1310720},{name:"L10.mlp.down.scales",offset:79298560,bytes:327680},{name:"L11.conv.in_proj.quant",offset:79626240,bytes:1572864},{name:"L11.conv.in_proj.scales",offset:81199104,bytes:393216},{name:"L11.conv.out_proj.quant",offset:81592320,bytes:524288},{name:"L11.conv.out_proj.scales",offset:82116608,bytes:131072},{name:"L11.mlp.gate_up.quant",offset:82247680,bytes:2621440},{name:"L11.mlp.gate_up.scales",offset:84869120,bytes:655360},{name:"L11.mlp.down.quant",offset:85524480,bytes:1310720},{name:"L11.mlp.down.scales",offset:86835200,bytes:327680},{name:"L12.attn.qkv.quant",offset:87162880,bytes:1048576},{name:"L12.attn.qkv.scales",offset:88211456,bytes:262144},{name:"L12.attn.o_proj.quant",offset:88473600,bytes:524288},{name:"L12.attn.o_proj.scales",offset:88997888,bytes:131072},{name:"L12.mlp.gate_up.quant",offset:89128960,bytes:2621440},{name:"L12.mlp.gate_up.scales",offset:91750400,bytes:655360},{name:"L12.mlp.down.quant",offset:92405760,bytes:1310720},{name:"L12.mlp.down.scales",offset:93716480,bytes:327680},{name:"L13.conv.in_proj.quant",offset:94044160,bytes:1572864},{name:"L13.conv.in_proj.scales",offset:95617024,bytes:393216},{name:"L13.conv.out_proj.quant",offset:96010240,bytes:524288},{name:"L13.conv.out_proj.scales",offset:96534528,bytes:131072},{name:"L13.mlp.gate_up.quant",offset:96665600,bytes:2621440},{name:"L13.mlp.gate_up.scales",offset:99287040,bytes:655360},{name:"L13.mlp.down.quant",offset:99942400,bytes:1310720},{name:"L13.mlp.down.scales",offset:101253120,bytes:327680},{name:"head.quant",offset:101580800,bytes:33554432},{name:"head.scales",offset:135135232,bytes:8388608},{name:"head.zero_points",offset:143523840,bytes:1048576},{name:"embed.quant",offset:101580800,bytes:33554432},{name:"embed.scales",offset:135135232,bytes:8388608},{name:"embed.zero_points",offset:143523840,bytes:1048576},{name:"L0.conv.in_proj.norm",offset:144572416,bytes:4096},{name:"L0.conv.core.taps",offset:144576512,bytes:12288},{name:"L0.mlp.gate_up.norm",offset:144588800,bytes:4096},{name:"L1.conv.in_proj.norm",offset:144592896,bytes:4096},{name:"L1.conv.core.taps",offset:144596992,bytes:12288},{name:"L1.mlp.gate_up.norm",offset:144609280,bytes:4096},{name:"L2.attn.qkv.norm",offset:144613376,bytes:4096},{name:"L2.attn.qkv.q_norm",offset:144617472,bytes:256},{name:"L2.attn.qkv.k_norm",offset:144617728,bytes:256},{name:"L2.mlp.gate_up.norm",offset:144617984,bytes:4096},{name:"L3.conv.in_proj.norm",offset:144622080,bytes:4096},{name:"L3.conv.core.taps",offset:144626176,bytes:12288},{name:"L3.mlp.gate_up.norm",offset:144638464,bytes:4096},{name:"L4.attn.qkv.norm",offset:144642560,bytes:4096},{name:"L4.attn.qkv.q_norm",offset:144646656,bytes:256},{name:"L4.attn.qkv.k_norm",offset:144646912,bytes:256},{name:"L4.mlp.gate_up.norm",offset:144647168,bytes:4096},{name:"L5.conv.in_proj.norm",offset:144651264,bytes:4096},{name:"L5.conv.core.taps",offset:144655360,bytes:12288},{name:"L5.mlp.gate_up.norm",offset:144667648,bytes:4096},{name:"L6.attn.qkv.norm",offset:144671744,bytes:4096},{name:"L6.attn.qkv.q_norm",offset:144675840,bytes:256},{name:"L6.attn.qkv.k_norm",offset:144676096,bytes:256},{name:"L6.mlp.gate_up.norm",offset:144676352,bytes:4096},{name:"L7.conv.in_proj.norm",offset:144680448,bytes:4096},{name:"L7.conv.core.taps",offset:144684544,bytes:12288},{name:"L7.mlp.gate_up.norm",offset:144696832,bytes:4096},{name:"L8.attn.qkv.norm",offset:144700928,bytes:4096},{name:"L8.attn.qkv.q_norm",offset:144705024,bytes:256},{name:"L8.attn.qkv.k_norm",offset:144705280,bytes:256},{name:"L8.mlp.gate_up.norm",offset:144705536,bytes:4096},{name:"L9.conv.in_proj.norm",offset:144709632,bytes:4096},{name:"L9.conv.core.taps",offset:144713728,bytes:12288},{name:"L9.mlp.gate_up.norm",offset:144726016,bytes:4096},{name:"L10.attn.qkv.norm",offset:144730112,bytes:4096},{name:"L10.attn.qkv.q_norm",offset:144734208,bytes:256},{name:"L10.attn.qkv.k_norm",offset:144734464,bytes:256},{name:"L10.mlp.gate_up.norm",offset:144734720,bytes:4096},{name:"L11.conv.in_proj.norm",offset:144738816,bytes:4096},{name:"L11.conv.core.taps",offset:144742912,bytes:12288},{name:"L11.mlp.gate_up.norm",offset:144755200,bytes:4096},{name:"L12.attn.qkv.norm",offset:144759296,bytes:4096},{name:"L12.attn.qkv.q_norm",offset:144763392,bytes:256},{name:"L12.attn.qkv.k_norm",offset:144763648,bytes:256},{name:"L12.mlp.gate_up.norm",offset:144763904,bytes:4096},{name:"L13.conv.in_proj.norm",offset:144768000,bytes:4096},{name:"L13.conv.core.taps",offset:144772096,bytes:12288},{name:"L13.mlp.gate_up.norm",offset:144784384,bytes:4096},{name:"head.norm",offset:144788480,bytes:4096},{name:"rope.cos",offset:144792576,bytes:131072},{name:"rope.sin",offset:144923648,bytes:131072},{name:"L0.mlp.down.kquant",offset:145054720,bytes:1310720},{name:"L0.mlp.down.kscales",offset:146365440,bytes:327680},{name:"L1.mlp.down.kquant",offset:146693120,bytes:1310720},{name:"L1.mlp.down.kscales",offset:148003840,bytes:327680},{name:"L2.mlp.down.kquant",offset:148331520,bytes:1310720},{name:"L2.mlp.down.kscales",offset:149642240,bytes:327680},{name:"L3.mlp.down.kquant",offset:149969920,bytes:1310720},{name:"L3.mlp.down.kscales",offset:151280640,bytes:327680},{name:"L4.mlp.down.kquant",offset:151608320,bytes:1310720},{name:"L4.mlp.down.kscales",offset:152919040,bytes:327680},{name:"L5.mlp.down.kquant",offset:153246720,bytes:1310720},{name:"L5.mlp.down.kscales",offset:154557440,bytes:327680},{name:"L6.mlp.down.kquant",offset:154885120,bytes:1310720},{name:"L6.mlp.down.kscales",offset:156195840,bytes:327680},{name:"L7.mlp.down.kquant",offset:156523520,bytes:1310720},{name:"L7.mlp.down.kscales",offset:157834240,bytes:327680},{name:"L8.mlp.down.kquant",offset:158161920,bytes:1310720},{name:"L8.mlp.down.kscales",offset:159472640,bytes:327680},{name:"L9.mlp.down.kquant",offset:159800320,bytes:1310720},{name:"L9.mlp.down.kscales",offset:161111040,bytes:327680},{name:"L10.mlp.down.kquant",offset:161438720,bytes:1310720},{name:"L10.mlp.down.kscales",offset:162749440,bytes:327680},{name:"L11.mlp.down.kquant",offset:163077120,bytes:1310720},{name:"L11.mlp.down.kscales",offset:164387840,bytes:327680},{name:"L12.mlp.down.kquant",offset:164715520,bytes:1310720},{name:"L12.mlp.down.kscales",offset:166026240,bytes:327680},{name:"L13.mlp.down.kquant",offset:166353920,bytes:1310720},{name:"L13.mlp.down.kscales",offset:167664640,bytes:327680},{name:"rope.cos_ext",offset:167992320,bytes:327680},{name:"rope.sin_ext",offset:168320000,bytes:327680}],maxQueries:1,maxPositions:2560,samplePartials:256,planPosition:1024,maxTotalPositions:2112,knee:[{shape:"attn_o_proj",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1}},{shape:"mlp_down",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1}},{shape:"attn_qkv",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1}},{shape:"conv_in_proj",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:4,subgroupsPerRow:1}},{shape:"mlp_gate_up",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1}},{shape:"head",geometry:{workgroupSize:128,subgroupSize:32,rowsPerSubgroup:4,subgroupsPerRow:1}}],dequantVariant:"carried",flashOptima:[{positions:192,blocks:16},{positions:1024,blocks:16},{positions:2048,blocks:16}],extraFeatures:["timestamp-query","shader-f16","subgroups"],powerPreference:"high-performance",expectedSubgroupSize:32},candidate:{key:"MIX_M40",flash:!0,unfused:!1,padding:0,flashMode:"subgroup",matvec:{attn_o_proj:{geometry:{workgroupSize:512,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1},variant:"v3_both"},conv_out_proj:{geometry:{workgroupSize:256,subgroupSize:32,rowsPerSubgroup:1,subgroupsPerRow:1},variant:"v3_both"},attn_qkv:{geometry:{workgroupSize:512,subgroupSize:32,rowsPerSubgroup:4,subgroupsPerRow:1},variant:"carried"},conv_in_proj:{geometry:{workgroupSize:512,subgroupSize:32,rowsPerSubgroup:6,subgroupsPerRow:1},variant:"carried"},head:{geometry:{workgroupSize:256,subgroupSize:32,rowsPerSubgroup:8,subgroupsPerRow:1},variant:"v3_both"}},mlpMode:"subgroup",mlpSlices:40,mlpWorkgroup:1024,normSubgroup:!0,convFuse:!0,flashOnline:!0,flashBlocks:32,mlpPair:!0,foldLanes:16,foldSubgroup:!0,mlpHalfDot:"gate",mlpHalfProduct:"down"}};var S1={prompt:[1,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,14367,521,14925,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,7410,779,2129,2076,5935,523,2357,779,4976,803,4843,779,3848,14118,988,779,31368,810,779,7980,10357,37370,1626,779,19814,521,810,968,4250,787,61604,779,27651,1320,8557],continuation:[523,941,1726,11099,1236,8366,779,2080,803,12546,896,1320,7410,779,2129,2076,5935,523,509,1098,1726,11099,6502,535,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,14367,521,14925,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,7410,779,2129,2076,5935,523,509,17978,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,14367,521,14925,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,7410,779,2129,2076,5935,523,509,17978,509,1098,1726,11099,6502,535,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,14367,521,14925,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,1320,7410,779,2129,2076,5935,523,509,17978,509,1098,1726,11099,6502,535,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,4915,521,29379,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,1320,7410,779,2129,2076,5935,523,509,17978,509,1098,1726,11099,6502,535,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,4915,521,14925,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,1320,7410,779,2129,2076,5935,523,509,17978,509,1098,1726,11099,6502,535,509,1098,809,43861,1726,11099,6502,797,1210,3446,4873,1973,4915,521,29379,779,7921,521,779,766,2259,521,810,779,6289,803,779,12546,896,1320,7410,779,2129,2076,5935,523,7,2,1,708,1098,2773,856,768,14500,803,779,2120,4520,803,779,4277,523,509,1098,4277,29617,779,7083,803,779,809,43861,797,779,36508,5294,523,509,1098,4277,11961,896,809,17307,17662,938,5175,875,20093,810,6155,797,779,5547,523,2216,3176,768,6260,7848,875,12546,811,28062,1149,1959,1892,811,23303,523,509,1098,4277,1236,22587,779,5160,3616,803,809,17307,17662,884,779,36508,5294,523,509,1098,4277,29617,779,3727,4063,803,809,17307,17662,810,1149,3338,4642,523,509,1098,4277,35283,896,779,2780,803,40686,810,14881,809,17307,17662,19968,9646,884,779,6249,810,779,2432,803,809,43861,523,509,1098,4277,1236,29617,779,5888,803,809,17307,17662,875,36508,6155,810,6368,523,509,1098,4277,35761,968,61498,779,7083,803,809,17307,17662,797,779,36508,5294,810,779,1595,875,6806,10961,797,1149,12003,810,1287,30132,523,7,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,3560,803,779,997,552,44413,14174,511,5116,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,3560,803,779,997,552,44413,14174,511,5116,797,779,5180,803,779,2704,3008,856,811,3176,2203,810,4768,4418,811,16310,3869,3609,810,7971,523,1470,5260,11552,1559,906,16310,24661,521,19290,521,6879,5180,521,810,8088,6925,523,941,5116,16070,811,1801,7039,2846,779,13207,803,16310,797,779,2704,3008,810,1352,3671,797,40402,779,3374,1090,9552,810,5009,523,14633,521,936,1400,3644,2203,884,16310,4072,521,3316,521,8022,521,810,1312,6879,5906,896,938,8611,811,779,2704,3008,523,941,6341,856,811,3755,768,12336,810,13914,7582,875,7039,9368,797,3481,1209,1337,16310,3869,810,5009,523,7,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,1947,803,779,997,552,44413,14174,511,797,779,2704,3008,856,811,3176,768,3255,803,6939,810,5139,3136,7712,521,5777,797,779,5180,803,3709,9552,810,6879,10607,523,1311,14428,906,768,12447,1685,779,2704,3008,810,779,16310,32182,521,51008,779,10607,803,6157,521,16954,521,810,6879,7971,523,1470,5260,779,1517,803,16310,26606,521,31931,2739,521,810,19656,896,938,7968,1593,797,2473,5009,521,1559,906,997,550,6217,768,16310,2512,997,550,6217,2473,2512,810,997,552,44413,2512,1144,938,16944,811,2473,9552,810,3129,523,14633,521,779,3869,14428,906,768,6690,875,17954,6879,5139,810,32588,521,63879,768,5033,803,9071,19504,810,2624,4079,3136,7712,810,8344,998,25848,523,3860,19623,916,16310,3869,810,5009,521,7712,1011,7425,19393,1419,779,4121,1535,803,779,16310,32182,1090,3129,810,779,14674,6879,15300,1685,779,1547,11502,523,7,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,1947,803,779,16310,14174,856,811,18655,6939,810,5139,1685,7712,810,8344,998,521,810,811,9679,6879,10607,810,23615,5139,523,1311,14428,906,768,12447,1685,779,1547,13225,521,24793,7712,811,13515,916,8344,998,797,15668,2105,810,811,21904,779,8345,2953,803,16310,5009,810,16954,523,941,16310,14174,856,1593,797,3727,5844,521,1559,906,18077,3869,521,4985,3869,521,810,6544,6939,521,810,10030,768,3069,3671,797,40402,779,6879,9552,810,4396,803,2059,11502,523,14633,521,936,856,1593,797,3727,5844,803,5075,810,3886,521,1559,906,16310,8587,521,21786,521,810,5936,521,811,19461,6879,19393,810,9679,5008,59352,5139,523,7,708,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,1947,803,779,16310,14174,856,811,18655,6939,810,5139,1685,7712,810,8344,998,521,810,811,9679,6879,10607,810,23615,5139,523,1311,14428,906,768,12447,1685,779,1547,13225,521,24793,7712,811,13515,916,8344,998,797,15668,2105,810,811,21904,779,8345,2953,803,16310,5009,810,16954,523,941,16310,14174,856,1593,797,3727,5844,521,1559,906,18077,3869,521,4985,3869,521,810,6544,6939,521,810,10030,768,3069,3671,797,40402,779,6879,9552,810,4396,803,2059,11502,523,1311,856,5175,875,7712,811,2111,810,21904,16310,5009,810,16954,811,27927,3020,6939,810,5139,1685,779,1547,4178,523,7,708,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,1947,803,779,16310,14174,856,811,18655,6939,810,5139,1685,7712,810,8344,998,521,810,811,9679,6879,10607,810,23615,5139,523,1311,14428,906,768,12447,1685,779,1547,13225,521,24793,7712,811,13515,916,8344,998,797,15668,2105,810,811,21904,779,8345,2953,803,16310,5009,810,16954,523,941,16310,14174,856,1593,797,3727,5844,521,1559,906,18077,3869,521,4985,3869,521,810,6544,6939,521,810,10030,768,3069,3671,797,40402,779,6879,9552,810,4396,803,2059,11502,523,1311,856,5175,875,7712,811,2111,810,21904,16310,5009,810,16954,811,27927,3020,6939,810,5139,1685,779,1547,4178,523,7,708,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803,779,2704,3008,1784,542,535,941,5410,1947,803,779,16310,14174,856,811,18655,6939,810,5139,1685,7712,810,8344,998,521,810,811,9679,6879,10607,810,23615,5139,523,1311,14428,906,768,12447,1685,779,1547,13225,521,24793,7712,811,13515,916,8344,998,797,15668,2105,810,811,21904,779,8345,2953,803,16310,5009,810,16954,523,941,16310,14174,856,1593,797,3727,5844,521,1559,906,18077,3869,521,4985,3869,521,810,6544,6939,521,810,10030,768,3069,3671,797,40402,779,6879,9552,810,4396,803,2059,11502,523,1311,856,5175,875,7712,811,2111,810,21904,16310,5009,810,16954,811,27927,3020,6939,810,5139,1685,779,1547,4178,523,7,708,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,558,535,3747,856,779,5410,1947,803,779,997,552,44413,14174,511,797,779,5180,803]};var a=(L,$={})=>postMessage({type:L,...$}),h2=()=>new Promise((L)=>setTimeout(L,0)),r0=async(L)=>Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",L)),($)=>$.toString(16).padStart(2,"0")).join(""),Q1=null,z0=null,i0=!1,u1=!1,K0=null,l1=!1,E0="WebGPU device";function V0(){let L=globalThis.__probeHooks;if(L.errors().length||L.lost().length)throw Error("The GPU reported an error. Unload the model and try again.")}async function u2(){if(!navigator.gpu)return{supported:!1,reason:"WebGPU is unavailable. Open this page in a recent Chrome or Edge browser on a supported computer."};let L=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!L)return{supported:!1,reason:"No WebGPU adapter is available in this browser."};let $=["shader-f16","subgroups"].filter((Z)=>!L.features.has(Z));if($.length)return{supported:!1,reason:`This decoder needs ${$.join(" and ")}. Your browser does not expose ${$.length===1?"this feature":"these features"}; the model has not been downloaded.`};let J=L.info;if(J.subgroupMinSize!==32||J.subgroupMaxSize!==32)return{supported:!1,reason:`This kernel is compiled for 32-lane subgroups. This adapter reports ${J.subgroupMinSize??"?"}–${J.subgroupMaxSize??"?"}. No compatible kernel is available here.`};if(L.limits.maxComputeInvocationsPerWorkgroup<1024||L.limits.maxComputeWorkgroupSizeX<1024)return{supported:!1,reason:"This GPU does not support the 1,024-thread workgroups used by MIX_M40. The model has not been downloaded."};if(L.limits.maxStorageBufferBindingSize<h1.runtime.weightBytes)return{supported:!1,reason:"This adapter cannot bind the 169 MB model buffer required by this decoder."};if(J.isFallbackAdapter)return{supported:!1,reason:"A hardware WebGPU adapter is required; a software adapter was detected."};return l1=L.features.has("timestamp-query"),E0=[J.description||J.vendor,J.architecture].filter(Boolean).join(" · ")||"WebGPU hardware adapter",{supported:!0,device:E0,timestamps:l1,subgroup:32}}async function q4(){let L=await fetch("/monarch/model-manifest.json",{signal:K0.signal});if(!L.ok)throw Error("The model manifest could not be loaded.");let $=await L.json();if($.weightSha256!=="ca1fea89fd9f3ca7e5d6d5720705c96f457a92edd0cfc811cf8d9391ac784cb0"||$.weightBytes!==h1.runtime.weightBytes)throw Error("The published model manifest does not match the tested decoder.");let J=new Uint8Array($.weightBytes),Z=0,X=null;try{X=await caches.open("monarch-weights-ca1fea89-v1")}catch{}for(let Y of $.chunks){if(u1)throw new DOMException("Stopped","AbortError");let Q=null,H=!1,j=await X?.match(Y.url).catch(()=>{return});if(j){let V=await j.arrayBuffer();if(V.byteLength===Y.bytes&&await r0(V)===Y.sha256)Q=V,H=!0;else await X?.delete(Y.url)}if(!Q){let V=await fetch(Y.url,{signal:K0.signal});if(!V.ok||!V.body)throw Error(`Model download failed (${V.status}). Try loading again.`);let G=new Uint8Array(Y.bytes),I=V.body.getReader(),D=0,R=0;while(!0){let{value:z,done:T}=await I.read();if(T)break;if(D+z.byteLength>G.byteLength)throw Error("Unexpected model chunk size.");if(G.set(z,D),D+=z.byteLength,performance.now()-R>80)a("progress",{stage:"Downloading weights",loaded:Z+D,total:$.weightBytes}),R=performance.now()}if(Q=G.buffer,D!==Y.bytes||await r0(Q)!==Y.sha256)throw Error("Model integrity check failed. Please retry the download.");try{await X?.put(Y.url,new Response(Q,{headers:{"Content-Type":"application/octet-stream"}}))}catch{}}J.set(new Uint8Array(Q),Z),Z+=Q.byteLength,a("progress",{stage:H?"Reading verified cached weights":"Downloading weights",loaded:Z,total:$.weightBytes})}if(Z!==$.weightBytes||await r0(J.buffer)!==$.weightSha256)throw Error("The complete model checksum did not match.");return J}async function A4(){let L=await u2();if(a("support",L),!L.supported)return;K0=new AbortController,u1=!1;let $={...h1.runtime,extraFeatures:h1.runtime.extraFeatures.filter((j)=>j!=="timestamp-query"||l1)};a("progress",{stage:"Preparing your GPU",loaded:0,total:h1.runtime.weightBytes}),await M0(d0($),"high-performance",32);let J=await q4();a("progress",{stage:"Loading tokenizer",loaded:J.length,total:J.length});let[Z,X]=await Promise.all(["tokenizer.json","tokenizer_config.json"].map(async(j)=>{let V=await fetch("/monarch/"+j,{signal:K0.signal});if(!V.ok)throw Error("Tokenizer download failed.");return V.json()}));z0=new g2(Z,X),O7([h1.candidate]),a("progress",{stage:"Compiling the 72-dispatch decoder",loaded:J.length,total:J.length});let Y=URL.createObjectURL(new Blob([J]));try{Q1=(await I2({...$,weightsUrl:Y})).session}finally{URL.revokeObjectURL(Y)}Q1.runtime.setArm("MIX_M40"),V0(),a("progress",{stage:"Checking a known 32-token continuation",loaded:J.length,total:J.length}),await I0(Q1,{tokenIds:Uint32Array.from(S1.prompt),promptLength:S1.prompt.length,seed:111008,freerun:!0});let Q=new Uint32Array(await Z0(Q1.gpu.device,Q1.runtime.buffers.sampled,(S1.prompt.length-1)*4,4))[0],H=await P0(32,S1.prompt.length);if(Q!==S1.continuation[0]||H.sampled.some((j,V)=>j!==S1.continuation[V+1]))throw Error("The decoder failed its known-answer check on this GPU. This device is not supported by the current kernel.");V0(),a("loaded",{device:E0,timestamps:l1,verifiedTokens:32,dispatches:Q1.runtime.dispatchCount,weightBytes:h1.runtime.weightBytes})}function P0(L,$){return B2(Q1,{tokens:L,startPosition:$,startStep:$,readbackEvery:32,ringTokens:64,timestamps:l1,freerun:!0,seed:111008})}function t0(L){return L.length?z0.decode(L,{skip_special_tokens:!0}):""}async function O4(L,$){if(!L.trim())throw Error("Enter a prompt first.");let J=`<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
${L.trim()}<|im_end|>
<|im_start|>assistant
`,Z=z0.encode(J,{add_special_tokens:!1}).ids;if(Z.length>1024)throw Error(`Your prompt is ${Z.length.toLocaleString()} tokens. Shorten it to fit the demo’s 1,024-token prompt limit.`);let X=z0.token_to_id("<|im_end|>"),Y=[],Q=performance.now();a("generating",{promptTokens:Z.length}),await I0(Q1,{tokenIds:Uint32Array.from(Z),promptLength:Z.length,seed:111008,freerun:!0});let H=new Uint32Array(await Z0(Q1.gpu.device,Q1.runtime.buffers.sampled,(Z.length-1)*4,4))[0],j=performance.now(),V=j-Q,G=0,I=0;if(H!==X)Y.push(H);a("text",{text:t0(Y),tokens:Y.length,tps:null,ttft:V});let D=H===X,R=Z.length;while(!u1&&!D&&Y.length<$){let z=await P0(32,R);R+=32,G+=32,I+=Math.max(0,z.gpuBusyMs);for(let N of z.sampled){if(N===X){D=!0;break}if(Y.length>=$)break;Y.push(N)}V0();let T=performance.now()-j;a("text",{text:t0(Y),tokens:Y.length,tps:Y.length>1?(Y.length-1)*1000/T:null,ttft:V,gpuTps:l1?G*1000/I:null}),await h2()}V0(),a("generated",{text:t0(Y),tokens:Y.length,tps:Y.length>1?(Y.length-1)*1000/(performance.now()-j):null,ttft:V,stopped:u1,promptTokens:Z.length,computedTokens:G+1})}async function F4(L){if(![192,1024].includes(L))throw Error("Unsupported benchmark context.");let $=[...S1.prompt,...S1.continuation],J=L-128,Z=[],X=()=>I0(Q1,{tokenIds:Uint32Array.from($.slice(0,J)),promptLength:J,seed:111008,freerun:!0});a("benchmarkProgress",{completed:0,total:4,stage:"Warming the decoder"}),await X(),await P0(64,J);for(let H=0;H<4&&!u1;H++){await X(),a("benchmarkProgress",{completed:H,total:4,stage:`Measuring run ${H+1} of 4`});let j=await P0(256,J);if(V0(),Z.length&&j.sampled.some((V,G)=>V!==Z[0].ids[G]))throw Error("Benchmark continuations differed between repeats. No valid result is reported.");Z.push({wallMs:j.wallMs,tps:256000/j.wallMs,gpuBusyMs:l1?j.gpuBusyMs:null,ids:j.sampled}),a("benchmarkProgress",{completed:H+1,total:4,stage:`Run ${H+1}: ${(256000/j.wallMs).toFixed(0)} tok/s`,tps:256000/j.wallMs}),await h2()}if(u1){a("cancelled");return}let Y=Z.map((H)=>H.wallMs/256).sort((H,j)=>H-j),Q=(Y[1]+Y[2])/2;a("benchmarkResult",{depth:L,tokens:256,runs:4,ms:Q,tps:1000/Q,runTps:Z.map((H)=>H.tps),spreadPct:100*(Y[3]-Y[0])/Q,device:E0,measuredAt:new Date().toISOString(),timestamps:l1,exactRepeats:!0,protocol:"4 × 256 free-running tokens; 64-token warmup; prefill excluded; 32-token readback through a 64-token ring; wall timer includes encoding and queue completion. Context is the center of the 256-token window. This quick browser run is not the original controlled lab confirmation."})}onmessage=async({data:L})=>{if(L.type==="cancel"){u1=!0,K0?.abort();return}if(L.type==="support"){try{a("support",await u2())}catch($){a("support",{supported:!1,reason:String($)})}return}if(i0)return;i0=!0,u1=!1;try{if(L.type==="load")await A4();else if(L.type==="generate"&&Q1)await O4(String(L.prompt??""),Math.min(512,Math.max(32,Number(L.limit)||256)));else if(L.type==="benchmark"&&Q1)await F4(Number(L.depth))}catch($){a("error",{message:$ instanceof Error?$.message:String($),fatal:!Q1||Boolean(globalThis.__probeHooks.lost().length)})}finally{i0=!1}};
