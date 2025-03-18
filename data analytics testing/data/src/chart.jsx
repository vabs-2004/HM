import React from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { name: "Training", value: 80, color: "#007bff" },
  { name: "Validation", value: 10, color: "#ff7f0e" },
  { name: "Testing", value: 10, color: "#7f7f7f" },
];

const DonutChart = ({ text }) => {
    // useEffect(() => {
    //     axios.get("https://api.example.com/chart-data") // API returning JSON
    //       .then((response) => {
    //         const jsonData = response.data; // Example: { "Training": 80, "Validation": 10, "Testing": 10 }
    
    //         // Convert JSON object to array of { name, value } objects
    //         const chartData = Object.keys(jsonData).map((key) => ({
    //           name: key,
    //           value: jsonData[key],
    //         }));
    
    //         setData(chartData);
    //         setLoading(false);
    //       })
    //       .catch((err) => {
    //         setError("Failed to fetch data");
    //         setLoading(false);
    //       });
    //   }, []);
  return (
    <div style={{ width: 250, height: 250, position: "relative", margin: "auto" }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={50} // Makes it a donut chart
            outerRadius={80} // Adjust size
            paddingAngle={2}
            dataKey="value"
            label={({ name }) => name}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          fontSize: "18px",
          fontWeight: "bold",
        }}
      >
        {text}
      </div>
    </div>
  );
};

export default DonutChart;
