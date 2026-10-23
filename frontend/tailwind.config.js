import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const glob = value => value.replaceAll('\\','/');
export default {content:[glob(path.join(root,'index.html')),glob(path.join(root,'src/**/*.{js,jsx}'))],theme:{extend:{colors:{forest:'#1d4937',lime:'#c9ef72',paper:'#f6f7f2'}}},plugins:[]};
