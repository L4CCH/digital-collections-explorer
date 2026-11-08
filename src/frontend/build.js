#!/usr/bin/env node

const { execSync } = require('child_process');
const { choiceOf, frontendDir, readConfig } = require('./frontends');

const { frontend, media } = choiceOf(readConfig());
const dir = frontendDir(frontend);

console.log(`Building the ${frontend} frontend, showing photos${media.map((name) => `, ${name}`).join('')}`);
try {
  execSync('npm run build', { cwd: dir, stdio: 'inherit' });
} catch (error) {
  console.error(`The build failed: ${error.message}`);
  process.exit(1);
}
