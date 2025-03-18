import React from "react";
import DonutChart from "./chart";

const App = () => {
  return (
    <div style={{ display: "flex", gap: "20px", justifyContent: "center", alignItems: "center", height: "100vh" }}>
      <DonutChart text="(a)" />
      <DonutChart text="(b)" />
      <DonutChart text="(c)" />
      <DonutChart text="(a)" />
      <DonutChart text="(b)" />
      <DonutChart text="(c)" />
    </div>
  );
};

export default App;
