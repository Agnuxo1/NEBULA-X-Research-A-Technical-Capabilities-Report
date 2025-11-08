
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PerformanceValidation } from '../types';

interface CompetitionChartProps {
  data: PerformanceValidation[];
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const rank = data.rank === 'Unranked' ? 'Unranked' : `Rank ${data.rank}`;
    return (
      <div className="bg-gray-800 text-white p-3 rounded-lg border border-gray-600 shadow-lg">
        <p className="font-bold text-indigo-400">{label}</p>
        <p>{`${rank} out of ${data.totalTeams} teams`}</p>
      </div>
    );
  }
  return null;
};

const CompetitionChart: React.FC<CompetitionChartProps> = ({ data }) => {
  const chartData = data
    .filter(item => typeof item.rank === 'number')
    .map(item => ({
      ...item,
      performance: (1 - (item.rank as number) / item.totalTeams) * 100, // Higher is better
    }))
    .sort((a, b) => b.performance - a.performance);

  return (
    <div className="w-full h-96">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{
            top: 5,
            right: 20,
            left: -10,
            bottom: 5,
          }}
          layout="vertical"
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
          <XAxis type="number" stroke="#A0AEC0" domain={[0, 100]} unit="%" />
          <YAxis dataKey="competition" type="category" width={150} stroke="#A0AEC0" fontSize={12} interval={0} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(128, 90, 213, 0.1)' }} />
          <Legend formatter={(value) => <span className="text-gray-300">{value}</span>} />
          <Bar dataKey="performance" name="Percentile Rank" fill="url(#colorUv)" />
          <defs>
            <linearGradient id="colorUv" x1="0" y1="0" x2="1" y2="0">
              <stop offset="5%" stopColor="#818CF8" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#A78BFA" stopOpacity={0.9}/>
            </linearGradient>
          </defs>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CompetitionChart;
