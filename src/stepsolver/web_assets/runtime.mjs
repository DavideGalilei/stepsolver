// Server-backed solver client used by the FastAPI application.

export function createSolverClient() {
  return {
    async solve(payload) {
      const response = await fetch("./api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail ?? "The server could not solve this query.");
      }
      return result;
    }
  };
}
