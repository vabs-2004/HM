import React from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";



const DonutChart = ({ text, data }) => {
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
            innerRadius={40} // Makes it a donut chart
            outerRadius={60} // Adjust size
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
          fontSize: "10px",
          fontWeight: "bold",
        }}
      >
        {text}
      </div>
    </div>
  );
};

export default DonutChart;
