import React, { useEffect, useState } from "react";
import "./css/App.css";
import Chart from "react-google-charts";
import { GoogleChartTicks } from "react-google-charts/dist/types";
// import { Scatter } from "react-chartjs-2";
import {
  CartesianGrid,
  LineChart,
  Tooltip,
  XAxis,
  Line,
  Scatter,
  Dot,
} from "recharts";
import { Ticks } from "chart.js";

type CircadianRythmMetrics = {
  min_morning: string;
  average_morning: string;
  max_morning: string;
  min_night: string;
  average_night: string;
  max_night: string;
};

type ApiData = {
  microwave_stovetop: {
    microwave: string;
    stovetop: string;
  };
  circadian_rythm_metrics: CircadianRythmMetrics;
  circadian_rythm_test: { morning: string; night: string };
};

function dateToString(date: Date) {
  return `${date.getHours()}:${date.getMinutes()}`;
}

function genTicks(data: ApiData) {
  const ticks = [];
  const [hour0, min0, sec0] = data.circadian_rythm_test.morning
    .split(":")
    .map((item) => parseInt(item));
  const test_morning_date = new Date(2021, 1, 1, hour0, min0, sec0);

  const [hour, min, sec] = data.circadian_rythm_metrics.min_morning
    .split(":")
    .map((item) => parseInt(item));
  const min_morning_date = new Date(2021, 1, 1, hour, min, sec);

  const [hour2, min2, sec2] = data.circadian_rythm_metrics.max_morning
    .split(":")
    .map((item) => parseInt(item));
  const max_morning_date = new Date(2021, 1, 1, hour2, min2, sec2);

  var oneMin = 1000 * 60;
  ticks.push(dateToString(new Date(min_morning_date.getTime())));
  for (
    var i = min_morning_date.getTime();
    i <= max_morning_date.getTime();
    i = i + oneMin
  ) {
    var rowDate = new Date(i);
    if (rowDate.getMinutes() === 0 || rowDate.getMinutes() === 30) {
      ticks.push(dateToString(rowDate));
    }
  }
  ticks.push(dateToString(new Date(max_morning_date.getTime())));
  return ticks;
}

function genCircadianRhythmTimeData(data: ApiData) {
  const res = [];
  const [hour0, min0, sec0] = data.circadian_rythm_test.morning
    .split(":")
    .map((item) => parseInt(item));
  const test_morning_date = new Date(2021, 1, 1, hour0, min0, sec0);

  const [hour, min, sec] = data.circadian_rythm_metrics.min_morning
    .split(":")
    .map((item) => parseInt(item));
  const min_morning_date = new Date(2021, 1, 1, hour, min, sec);

  const [hour2, min2, sec2] = data.circadian_rythm_metrics.max_morning
    .split(":")
    .map((item) => parseInt(item));
  const max_morning_date = new Date(2021, 1, 1, hour2, min2, sec2);

  var oneMin = 1000 * 60;
  for (
    var i = min_morning_date.getTime();
    i <= max_morning_date.getTime();
    i = i + oneMin
  ) {
    var rowDate = new Date(i);
    if (rowDate.getTime() === test_morning_date.getTime()) {
      res.push({ name: dateToString(rowDate), uv: 0 });
    } else {
      res.push({ name: dateToString(rowDate) });
    }
  }
  return res;
}

function App() {
  const [data, setData] = useState<ApiData>(Object);
  const [circadianRhythmTimeData, setCircadianRhythmData] = useState<Object[]>(
    []
  );
  const [ticks, setTicks] = useState<string[]>([]);

  useEffect(() => {
    fetch("http://localhost:3001/peaks")
      .then((response) => {
        if (response.ok) {
          return response.json();
        }
        throw response;
      })
      .then((data) => {
        setData(data);
        console.log("DATA: ", data);
        setCircadianRhythmData(genCircadianRhythmTimeData(data));
        console.log("CIRCRY: ", circadianRhythmTimeData);
        setTicks(genTicks(data));
        console.log("TICKS: ", ticks);
      })
      .catch((error) => {
        console.log("Error fetching data: ", error);
      });
  }, []);

  return (
    <div className="App">
      <h1 style={{ textAlign: "center" }}>Visualisations</h1>
      {data.microwave_stovetop ? (
        <Chart
          width={"600px"}
          height={"400px"}
          chartType="ScatterChart"
          loader={<div>Loading Chart</div>}
          data={[
            ["Microwave Use", "Stovetop Use"],
            [
              data.microwave_stovetop.microwave,
              data.microwave_stovetop.stovetop,
            ],
          ]}
          options={{
            title: "Microwave vs Stovetop Use",
            hAxis: {
              title: "Microwave Use (x)",
              minValue: -1,
              maxValue: 1,
              gridlines: { color: "#FFF" },
              textPosition: "none",
            },
            vAxis: {
              title: "Stovetop Use (y)",
              minValue: -1,
              maxValue: 1,
              gridlines: { color: "#FFF" },
              textPosition: "none",
            },
            legend: "none",
          }}
          rootProps={{ "data-testid": "1" }}
        />
      ) : (
        <p>Fetching microwave_stovetop data from API</p>
      )}
      {/* <p>{JSON.stringify(data.microwave_stovetop)}</p> */}

      {data.circadian_rythm_metrics && data.circadian_rythm_test ? (
        // <Scatter type="scatter" data={data} />
        <LineChart
          width={500}
          height={100}
          data={circadianRhythmTimeData}
          margin={{
            top: 5,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <XAxis dataKey="name" ticks={ticks} />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="uv"
            stroke="red"
            fill="red"
            strokeWidth="2"
            isAnimationActive={false}
          />
        </LineChart>
      ) : (
        // <Chart
        //   width={"600px"}
        //   height={"200px"}
        //   chartType="ScatterChart"
        //   loader={<div>Loading Chart</div>}
        //   data={[
        //     ["x", "y"],
        //     [0, 0],
        //   ]}
        //   options={{
        //     title: "Circadian Rhythm",
        //     hAxis: {
        //       title: "",
        //       ticks: circadianRhthymData,
        //       format: "H/m/s",
        //       // ticks: Object.values(data.circadian_rythm_metrics).map((time) => {
        //       //   console.log(time);
        //       //   var date = new Date(
        //       //     2021,
        //       //     5,
        //       //     7,
        //       //     parseInt(time.substring(0, 2)),
        //       //     parseInt(time.substring(3, 5)),
        //       //     parseInt(time.substring(6, 8))
        //       //   );
        //       //   return date;
        //       // }),
        //       // ticks: getDates()[1],
        //       // minValue: -1,
        //       // maxValue: 1,
        //       gridlines: { color: "#FFF" },
        //       baselineColor: "#FFF",
        //     },
        //     vAxis: {
        //       title: "",
        //       minValue: 0,
        //       maxValue: 0,
        //       gridlines: { color: "#FFF" },
        //       textPosition: "none",
        //     },
        //     legend: "none",
        //   }}
        //   rootProps={{ "data-testid": "1" }}
        // />
        <p>Fetching circadian rythym data from API</p>
      )}
      {/* <p>{JSON.stringify(data.circadian_rythm_metrics)}</p> */}
      {/* <p>{JSON.stringify(data.circadian_rythm_test)}</p> */}
    </div>
  );
}

export default App;

// import "./styles.css";
// import React from "react";
// import { LineChart, Line, XAxis, CartesianGrid, Tooltip } from "recharts";

// const data = [
//   {
//     name: "5:13",
//   },
//   {
//     name: "5:30",
//   },
//   {
//     name: "6:00",
//   },
//   {
//     name: "6:30",
//     uv: 0,
//   },
//   {
//     name: "7:00",
//   },
//   {
//     name: "7:30",
//   },
//   {
//     name: "8:00",
//   },
// ];

// export default function App() {
//   return (
//     <LineChart
//       width={500}
//       height={100}
//       data={data}
//       margin={{
//         top: 500,
//         right: 30,
//         left: 20,
//         bottom: 5,
//       }}
//     >
//       <CartesianGrid strokeDasharray="3 3" />
//       <XAxis dataKey="name" />
//       <Tooltip />
//       <Line type="monotone" dataKey="uv" stroke="red" fill="red" />
//     </LineChart>
//   );
// }
