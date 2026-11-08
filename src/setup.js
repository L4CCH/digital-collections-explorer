#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { execSync } = require('child_process');
const chalk = require('chalk');
const {
  CONFIG_PATH,
  DEFAULT_FRONTEND,
  LEGACY_TYPES,
  MEDIA,
  ROOT,
  choiceOf,
  dependenciesOf,
  frontendDir,
  readConfig,
  viewerDir,
  writeConfig,
} = require('./frontend/frontends');

const HELP = `Usage: npm run setup -- [options]

Builds the site's frontend. Photos and other images are always shown.
Add what else the collection holds, and only those packages are installed.

Options:
  --with=<kinds>         also show these, separated by commas, or none:
${Object.entries(MEDIA).map(([name, { label }]) => `                           ${name.padEnd(5)} ${label}`).join('\n')}
  --frontend=<name>      build another frontend: a folder in src/frontend,
                         or a path to one (default: ${DEFAULT_FRONTEND})
  --help, -h             show this help

Without --with, setup asks, or keeps the choice already in config.json.`;

const args = process.argv.slice(2);
const option = (name) => {
  const arg = args.find((value) => value.startsWith(`--${name}=`));
  return arg === undefined ? undefined : arg.slice(name.length + 3);
};

const fail = (message, hint) => {
  console.error(chalk.red(`Error: ${message}`));
  if (hint) {
    console.log(chalk.yellow(hint));
  }
  process.exit(1);
};

const parseMedia = (value) => {
  const names = value.split(',').map((name) => name.trim().toLowerCase()).filter(Boolean);
  const media = names.filter((name) => name !== 'none');
  const unknown = media.filter((name) => !MEDIA[name]);
  if (unknown.length) {
    fail(`Unknown kind: ${unknown.join(', ')}`, `Choose from: ${Object.keys(MEDIA).join(', ')}, or none`);
  }
  return [...new Set(media)];
};

const ask = (question) => new Promise((resolve) => {
  const prompt = readline.createInterface({ input: process.stdin, output: process.stdout });
  prompt.question(question, (answer) => {
    prompt.close();
    resolve(answer.trim().toLowerCase());
  });
});

const askForMedia = async (current) => {
  console.log('Photos and other images are always shown. Does the collection also hold:');
  const chosen = [];
  for (const [name, { label }] of Object.entries(MEDIA)) {
    const isChosen = current.includes(name);
    const answer = await ask(`  ${label}? ${isChosen ? '[Y/n]' : '[y/N]'} `);
    if (answer ? answer.startsWith('y') : isChosen) {
      chosen.push(name);
    }
  }
  return chosen;
};

const chooseMedia = async (current) => {
  const listed = option('with');
  if (listed !== undefined) {
    return parseMedia(listed);
  }
  const type = option('type');
  if (type !== undefined) {
    if (!LEGACY_TYPES[type]) {
      fail(`Unknown collection type '${type}'`, 'Use --with instead. See npm run setup -- --help');
    }
    console.log(chalk.yellow(`--type is replaced by --with. Building with --with=${LEGACY_TYPES[type].join(',') || 'none'}\n`));
    return LEGACY_TYPES[type];
  }
  if (process.stdin.isTTY) {
    return askForMedia(current);
  }
  return current;
};

const run = (command, cwd) => {
  try {
    execSync(command, { cwd, stdio: 'inherit' });
  } catch (error) {
    fail(`${command} failed in ${path.relative(ROOT, cwd) || '.'}: ${error.message}`);
  }
};

const main = async () => {
  if (args.includes('--help') || args.includes('-h')) {
    console.log(HELP);
    return;
  }
  if (!fs.existsSync(CONFIG_PATH)) {
    fail(`Configuration file not found: ${CONFIG_PATH}`, 'Make sure you have a config.json file in the project root.');
  }

  const config = readConfig();
  const current = choiceOf(config);
  const frontend = option('frontend') || current.frontend;
  const dir = frontendDir(frontend);
  if (!fs.existsSync(path.join(dir, 'package.json'))) {
    fail(`No frontend found at ${dir}`, 'Clone a frontend into src/frontend/<name>, then run setup with --frontend=<name>.');
  }
  const isDefault = dir === frontendDir(DEFAULT_FRONTEND);

  console.log(chalk.blue('Digital Collections Explorer - Setup\n'));
  const media = await chooseMedia(current.media);

  const {
    collection_type: _type, frontend: _frontend, media: _media, ...rest
  } = config;
  writeConfig({ frontend, media, ...rest });
  console.log(chalk.green(`\n✓ config.json: frontend ${frontend}, showing photos${media.map((name) => `, ${name}`).join('')}`));

  console.log(chalk.gray(`\nInstalling the ${frontend} frontend...`));
  run('npm install', dir);
  if (isDefault) {
    media.filter((name) => dependenciesOf(viewerDir(name)).length).forEach((name) => {
      console.log(chalk.gray(`\nInstalling the ${name} viewer...`));
      run('npm install', viewerDir(name));
    });
  }

  console.log(chalk.gray('\nBuilding...'));
  run('npm run build', dir);
  console.log(chalk.green('\n✓ Setup completed successfully!'));

  const python = media.flatMap((name) => MEDIA[name].python);
  if (python.length) {
    console.log(chalk.yellow('\nIndexing these needs a few more Python packages:'));
    python.forEach((command) => console.log(`  ${command}`));
  }
  console.log(chalk.yellow('\nGo check README.md for the next step.'));
};

main();
