// Cross-platform launcher for `npm run dev`.
// Tries `python3` first (common on macOS/Linux), falls back to `python` (common on Windows).
const { spawnSync, spawn } = require("child_process");

function commandExists(cmd) {
  const result = spawnSync(cmd, ["--version"], { stdio: "ignore" });
  return result.status === 0;
}

const pythonCmd = commandExists("python3") ? "python3" : (commandExists("python") ? "python" : null);

if (!pythonCmd) {
  console.error("Could not find a Python interpreter (tried `python3` and `python`).");
  console.error("Please install Python 3 and ensure it is on your PATH.");
  process.exit(1);
}

console.log(`Starting Flask app with: ${pythonCmd} app.py`);
const child = spawn(pythonCmd, ["app.py"], { stdio: "inherit" });
child.on("exit", (code) => process.exit(code));
