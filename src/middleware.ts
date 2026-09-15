import {defineMiddleware} from 'astro:middleware';
export const onRequest=defineMiddleware(async({url},next)=>{
 const response=await next();
 if(url.pathname==='/experiments/monarch'||url.pathname==='/experiments/monarch/'){
  response.headers.set('Cross-Origin-Opener-Policy','same-origin');
  response.headers.set('Cross-Origin-Embedder-Policy','require-corp');
 }
 return response;
});
