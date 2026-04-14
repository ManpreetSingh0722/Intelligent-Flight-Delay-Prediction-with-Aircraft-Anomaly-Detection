import React, { useState, useEffect } from 'react';
import './index.css';

const API_URL = 'http://127.0.0.1:5000/api';

const NASA_PROBES = {
  'T24': 'Total Temp at LPC outlet',
  't24': 'Total Temp at LPC outlet',
  'T30': 'Total Temp at HPC outlet',
  't30': 'Total Temp at HPC outlet',
  'T48': 'Total Temp at HPT',
  't48': 'Total Temp at HPT',
  'T50': 'Total Temp at LPT',
  't50': 'Total Temp at LPT',
  'P2': 'Total Pressure at fan inlet',
  'p2': 'Total Pressure at fan inlet',
  'P15': 'Total Pressure at bypass',
  'p15': 'Total Pressure at bypass',
  'P21': 'Total Pressure at compressor',
  'p21': 'Total Pressure at compressor',
  'P24': 'Total Pressure at HPC',
  'p24': 'Total Pressure at HPC',
  'P30': 'Total Pressure at HPC outlet',
  'p30': 'Total Pressure at HPC outlet',
  'P40': 'Total Pressure at burner',
  'p40': 'Total Pressure at burner',
  'P50': 'Total Pressure at LPT',
  'p50': 'Total Pressure at LPT',
  'Ps30': 'Static Pressure at HPC outlet',
  'ps30': 'Static Pressure at HPC outlet',
  'NC': 'Physical core speed',
  'Nc': 'Physical core speed',
  'nc': 'Physical core speed',
  'NF': 'Physical fan speed',
  'Nf': 'Physical fan speed',
  'nf': 'Physical fan speed',
  'WF': 'Fuel flow',
  'Wf': 'Fuel flow',
  'wf': 'Fuel flow'
};

// Helper to generate realistic looking airport codes from encoded numbers
const AIRPORTS = ['JFK', 'LAX', 'ORD', 'LHR', 'HND', 'DXB', 'MIA', 'SFO', 'ATL', 'DFW', 'DEN', 'LAS', 'SEA', 'BOS', 'EWR', 'MCO', 'IAD'];
const numToAirport = (num) => {
  if (isNaN(num)) return num; // if it's already a string
  return AIRPORTS[Number(num) % AIRPORTS.length];
};

export default function App() {
  const [flights, setFlights] = useState([]);
  const [delayedFlights, setDelayedFlights] = useState([]);
  const [selectedFlight, setSelectedFlight] = useState(null);

  const [isPredictingDelay, setIsPredictingDelay] = useState(false);
  const [isPredictingSensor, setIsPredictingSensor] = useState(false);

  const [currentTime, setCurrentTime] = useState('');
  const [modelFeatures, setModelFeatures] = useState({});
  const [sensorFeatures, setSensorFeatures] = useState({});
  const [showArchitecture, setShowArchitecture] = useState(false);
  const [page, setPage] = useState(1); // Track Dataset Pagination

  const generateLinePath = (features) => {
    const keys = Object.keys(features);
    if (keys.length === 0) return "";
    const points = keys.map((k, i) => {
      const x = (i / (keys.length - 1)) * 1000;
      let val = Number(features[k]);
      if (isNaN(val)) val = 0;
      // map val (-4 to 4) to SVG Y (100 to 0) => Center is 50
      let y = 50 - (val * 12.5);
      y = Math.max(2, Math.min(98, y)); // Clamp slightly off edge
      return `${x},${y}`;
    });
    return `M ${points.join(' L ')}`;
  };

  useEffect(() => {
    const timer = setInterval(() => {
      const d = new Date();
      setCurrentTime(d.toLocaleTimeString('en-US', { hour12: false }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [flightRes, sensorRes] = await Promise.all([
          fetch(`${API_URL}/data/flight_samples?page=${page}&limit=200`),
          fetch(`${API_URL}/data/sensor_samples?page=${page}&limit=200`)
        ]);

        const flightData = await flightRes.json();
        const sensorData = await sensorRes.json();

        if (flightData.samples && flightData.samples.length > 0) {
          const mapped = flightData.samples.map((s, idx) => {
            // Assign a sensor sample to this flight if available
            const s_sample = (sensorData.samples && sensorData.samples[idx]) ? sensorData.samples[idx] : null;

            return {
              id: `FL-${idx + 1000 + ((page - 1) * 200)}`,
              originalData: s,
              sensorData: s_sample,
              origin: numToAirport(s.origin_airport || s.origin),
              destination: numToAirport(s.destination_airport || s.destination),
              carrier: s.carrier_code || `CR`,
              time: `${String((idx % 14) + 6).padStart(2, '0')}:${String((idx * 7) % 60).padStart(2, '0')}`,
              gate: `G${Math.floor(Math.random() * 40) + 1}`,

              delayStatus: 'STANDBY',
              delayColor: 'text-gray-500',
              delayPredicted: false,
              delayProb: null,

              sensorStatus: 'STANDBY',
              sensorColor: 'text-gray-500',
              sensorPredicted: false,
              sensorScore: null
            };
          });
          setFlights(mapped);
        }
      } catch (err) {
        console.error("API failed:", err);
      }
    };

    fetchData();
  }, [page]);

  const openModal = (flight) => {
    setSelectedFlight(flight);
    setModelFeatures({ ...flight.originalData });
    setSensorFeatures({ ...(flight.sensorData || {}) });
  };

  const handleFeatureChange = (key, val) => {
    setModelFeatures(prev => ({ ...prev, [key]: Number(val) }));
  };

  const handleSensorChange = (key, val) => {
    setSensorFeatures(prev => ({ ...prev, [key]: Number(val) }));
  };

  const handlePredictAll = async () => {
    setIsPredictingDelay(true);
    setIsPredictingSensor(true);

    let isDelayed = false;
    let delayProb = 0;

    let isAnomaly = false;
    let anomalyScore = 0;

    try {
      const delayRes = await fetch(`${API_URL}/predict/delay`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(modelFeatures)
      });
      const d_data = await delayRes.json();
      if (d_data.error) {
        alert("DELAY API ERROR:\n" + d_data.error);
        throw new Error(d_data.error);
      } else {
        isDelayed = d_data.delayed;
        delayProb = d_data.probability;
      }
    } catch (e) {
      console.error(e);
      isDelayed = null; // Mark as failed
    }

    setIsPredictingDelay(false);

    try {
      if (selectedFlight.sensorData) {
        const sensorRes = await fetch(`${API_URL}/predict/anomaly`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(sensorFeatures)
        });
        const s_data = await sensorRes.json();
        if (s_data.error) {
          alert("SENSOR API ERROR:\n" + s_data.error);
          throw new Error(s_data.error);
        } else {
          isAnomaly = s_data.is_anomaly;
          anomalyScore = s_data.anomaly_score;
        }
      } else {
        isAnomaly = null;
      }
    } catch (e) {
      console.error(e);
      isAnomaly = null;
    }

    setIsPredictingSensor(false);

    const dStatus = isDelayed === null ? 'ERROR' : (isDelayed ? 'DELAYED' : 'ON TIME');
    const dColor = isDelayed === null ? 'text-gray-500' : (isDelayed ? 'text-fids-red text-glow-red' : 'text-fids-green text-glow-green');

    const sStatus = isAnomaly === null ? 'ERROR' : (isAnomaly ? 'WARNING' : 'HEALTHY');
    const sColor = isAnomaly === null ? 'text-gray-500' : (isAnomaly ? 'text-fids-red text-glow-red' : 'text-fids-green text-glow-green');

    const updatedFlight = {
      ...selectedFlight,
      delayStatus: dStatus,
      delayColor: dColor,
      delayPredicted: true,
      delayProb,
      sensorStatus: sStatus,
      sensorColor: sColor,
      sensorPredicted: true,
      sensorScore: anomalyScore
    };

    setSelectedFlight(updatedFlight);
    setFlights(prev => prev.map(f => f.id === updatedFlight.id ? updatedFlight : f));

    if (isDelayed === true || isAnomaly === true) {
      setDelayedFlights(prev => {
        if (!prev.find(p => p.id === updatedFlight.id)) return [...prev, updatedFlight];
        return prev.map(p => p.id === updatedFlight.id ? updatedFlight : p);
      });
    }
  };

  return (
    <div className="h-screen bg-[#0a0a0a] p-4 md:p-8 font-mono text-white flex flex-col relative overflow-hidden">
      {/* Background CRT Effects */}
      <div className="absolute inset-0 pointer-events-none before:content-[''] before:absolute before:inset-0 before:bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] before:opacity-20 z-0"></div>

      {/* Header */}
      <header className="flex-none relative z-10 flex flex-col md:flex-row justify-between items-start md:items-end border-b-2 border-[#333] pb-4 mb-6 text-fids-yellow">
        <div>
          <h1 className="text-4xl md:text-5xl font-black tracking-widest text-glow-yellow mb-2">DEPARTURES</h1>
          <p className="text-xs md:text-sm text-gray-400 tracking-[0.2em] uppercase">Flight Delay & Aircraft Sensor ML System</p>
        </div>
        <div className="text-right mt-4 md:mt-0 flex flex-col items-end">
          <div className="text-3xl md:text-4xl font-bold tracking-widest text-glow-yellow">{currentTime || '00:00:00'}</div>

          <div className="flex items-center gap-4 mt-2">
            <button
              onClick={() => setShowArchitecture(true)}
              className="text-xs tracking-[0.2em] border border-[#444] bg-[#111] px-3 py-1 rounded hover:bg-fids-yellow hover:text-black hover:border-fids-yellow transition-colors font-bold uppercase text-gray-300 pointer-events-auto"
            >
              ML ARCHITECTURE
            </button>
            <div className={`text-xs md:text-sm tracking-[0.2em] uppercase text-fids-green pointer-events-auto`}>
              SYSTEM ONLINE
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 min-h-0 flex flex-col xl:flex-row gap-6 relative z-10">
        <div className="flex-1 flex flex-col min-h-0">
          <div className="fids-board bg-fids-board p-0 text-sm md:text-base flex-1 flex flex-col overflow-hidden">
            {/* Headers - Fixed at top of flex container */}
            <div className="flex-none bg-[#0a0a0a] grid grid-cols-12 gap-2 p-4 border-b-2 border-[#333] text-gray-500 font-bold tracking-wider">
              <div className="col-span-2">FLIGHT</div>
              <div className="col-span-1">ORIG</div>
              <div className="col-span-1">DEST</div>
              <div className="col-span-1 text-center">TIME</div>
              <div className="col-span-1 text-center">GATE</div>
              <div className="col-span-3 text-right">DELAY AI</div>
              <div className="col-span-3 text-right pr-2">ENGINE SENSOR AI</div>
            </div>

            {/* Rows */}
            <div className="flex-1 overflow-y-auto pb-4 custom-scrollbar">
              {flights.map((flight) => (
                <div
                  key={flight.id}
                  onClick={() => openModal(flight)}
                  className="fids-row grid grid-cols-12 gap-2 py-3 px-4 items-center text-fids-yellow cursor-pointer hover:bg-[#1a1a1a]"
                >
                  <div className="col-span-2 font-bold tracking-wider text-lg">{flight.id}</div>
                  <div className="col-span-1 uppercase text-white font-bold">{flight.origin}</div>
                  <div className="col-span-1 uppercase text-white font-bold">{flight.destination}</div>
                  <div className="col-span-1 text-center font-bold">{flight.time}</div>
                  <div className="col-span-1 text-center text-gray-400">{flight.gate}</div>
                  <div className={`col-span-3 text-right font-bold text-lg ${flight.delayColor} ${flight.delayPredicted ? 'flicker-in' : ''}`}>
                    {flight.delayStatus}
                  </div>
                  <div className={`col-span-3 text-right pr-2 font-bold text-lg ${flight.sensorColor} ${flight.sensorPredicted ? 'flicker-in' : ''}`}>
                    {flight.sensorStatus}
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination Controls */}
            <div className="flex-none flex justify-between items-center p-4 border-t-2 border-[#333] bg-[#0a0a0a]">
              <button
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
                className="bg-fids-yellow text-black px-6 py-2 rounded font-bold uppercase tracking-widest disabled:opacity-50 hover:bg-yellow-400"
              >
                &larr; PREV 200 RUNWAYS
              </button>
              <div className="text-gray-500 font-bold tracking-widest">
                DATABASE BLOCK: {page} / 500+
              </div>
              <button
                onClick={() => setPage(p => p + 1)}
                className="bg-fids-yellow text-black px-6 py-2 rounded font-bold uppercase tracking-widest hover:bg-yellow-400"
              >
                NEXT 200 RUNWAYS &rarr;
              </button>
            </div>
          </div>
        </div>

        {/* Alerts Panel */}
        {delayedFlights.length > 0 && (
          <div className="xl:w-1/3 fids-board p-4 flex flex-col min-h-0 border-fids-red">
            <h2 className="flex-none text-fids-red text-glow-red text-2xl font-bold tracking-widest mb-4 pb-2 border-b-2 border-fids-red bg-[#0a0a0a]">
              ATTENTION: ALERTS
            </h2>
            <div className="flex-1 overflow-y-auto flex flex-col gap-4 pr-2 custom-scrollbar">
              {delayedFlights.map(df => (
                <div key={df.id} className="border border-fids-red p-3 rounded bg-[rgba(255,0,0,0.05)]">
                  <div className="font-bold text-fids-yellow text-lg mb-1">{df.id} <span className="text-gray-400 text-sm">({df.origin} &rarr; {df.destination})</span></div>
                  <div className="flex justify-between items-center text-sm">
                    <span className="text-gray-400">Delay Risk:</span>
                    <span className={df.delayColor}>{df.delayStatus} {df.delayProb !== null ? `(${Math.round(df.delayProb * 100)}%)` : ''}</span>
                  </div>
                  <div className="flex justify-between items-center text-sm mt-1">
                    <span className="text-gray-400">Sensor Health:</span>
                    <span className={df.sensorColor}>{df.sensorStatus} {df.sensorScore !== null ? `(${df.sensorScore.toFixed(2)})` : ''}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Interactive Testing Modal */}
      {selectedFlight && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 modal-overlay px-4 py-6">
          <div className="fids-board bg-black border-2 border-fids-yellow p-6 md:p-8 max-w-[95vw] lg:max-w-7xl w-full relative modal-content max-h-full flex flex-col overflow-hidden">
            <button
              className="absolute top-4 right-6 text-4xl text-gray-500 hover:text-white z-20"
              onClick={() => setSelectedFlight(null)}
            >
              ×
            </button>
            <h2 className="text-fids-yellow text-glow-yellow text-2xl lg:text-3xl font-bold border-b-2 border-[#333] pb-4 mb-6 uppercase tracking-widest flex flex-col md:flex-row items-start md:items-center gap-2 md:gap-4 flex-none pr-10">
              <span className="whitespace-nowrap">{selectedFlight.id}</span>
              <span className="text-gray-500 text-lg lg:text-xl font-normal hidden md:inline">AI DIAGNOSTICS & TELEMETRY</span>
            </h2>

            <div className="overflow-y-auto custom-scrollbar flex-1 pr-2 pb-6 block">
              {/* Quick Metrics */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-y-4 gap-x-8 text-xl mb-6 bg-[#0a0a0a] border border-[#222] p-4">
                <div>
                  <div className="text-gray-500 text-xs mb-1 uppercase tracking-wider">Flight Context</div>
                  <div className="text-white font-bold">{selectedFlight.origin} &rarr; {selectedFlight.destination}</div>
                </div>
                <div>
                  <div className="text-gray-500 text-xs mb-1 uppercase tracking-wider">Carrier</div>
                  <div className="text-white font-bold">{selectedFlight.carrier}</div>
                </div>
                <div>
                  <div className="text-gray-500 text-xs mb-1 uppercase tracking-wider">Random Forest Delay Risk</div>
                  <div className={`font-bold ${selectedFlight.delayColor}`}>
                    {selectedFlight.delayStatus} {selectedFlight.delayPredicted && `(${(selectedFlight.delayProb * 100).toFixed(1)}%)`}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 text-xs mb-1 uppercase tracking-wider">Isolation Forest Stability</div>
                  <div className={`font-bold ${selectedFlight.sensorColor}`}>
                    {selectedFlight.sensorStatus} {selectedFlight.sensorScore && `(${selectedFlight.sensorScore.toFixed(2)})`}
                  </div>
                </div>
              </div>

              <div className="mb-6 bg-[#0a0a0a] border border-[#222] p-5 relative overflow-hidden">
                <div className="text-glow-yellow text-fids-yellow text-xs mb-4 uppercase tracking-widest font-bold">Prediction Probability Graphs</div>

                {/* Delay Graph */}
                <div className="w-full bg-[#111] h-8 relative rounded overflow-hidden border border-[#333]">
                  {selectedFlight.delayPredicted ? (
                    <>
                      <div
                        className="absolute top-0 left-0 h-full bg-fids-red transition-all duration-1000 ease-in-out border-r border-[#111]"
                        style={{ width: `${selectedFlight.delayProb * 100}%` }}
                      ></div>
                      <div
                        className="absolute top-0 right-0 h-full bg-fids-green transition-all duration-1000 ease-in-out z-0"
                        style={{ width: `${(1 - selectedFlight.delayProb) * 100}%` }}
                      ></div>
                      <div className="absolute inset-0 flex justify-between items-center px-4 font-bold text-xs uppercase z-10 text-white pointer-events-none text-shadow-md">
                        <span>Delayed Confidence: {(selectedFlight.delayProb * 100).toFixed(1)}%</span>
                        <span>On-Time Confidence: {((1 - selectedFlight.delayProb) * 100).toFixed(1)}%</span>
                      </div>
                    </>
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center font-bold text-xs uppercase text-gray-600 tracking-widest pointer-events-none">
                      [ AWAITING DELAY PREDICTION EXECUTION ]
                    </div>
                  )}
                </div>

                {/* Sensor Graph */}
                <div className="mt-6 pt-4 border-t border-[#222]">
                  <div className="text-gray-500 text-xs mb-2 uppercase tracking-widest flex justify-between drop-shadow-md">
                    <span>Isolation Forest Distribution (Anomaly Score Graph)</span>
                    <span className="text-gray-600">Critical Cutoff: 0.0</span>
                  </div>
                  <div className="w-full bg-[#111] h-6 relative rounded overflow-hidden border border-[#333]">
                    {selectedFlight.sensorPredicted && selectedFlight.sensorScore !== null ? (
                      <>
                        {/* Gauge from -1 to +1. The center 50% mark represents 0 */}
                        <div className="absolute top-0 bottom-0 left-[20%] w-0.5 bg-gray-600 z-10" title="Threshold"></div>
                        <div
                          className={`absolute top-0 h-full transition-all duration-1000 ${selectedFlight.sensorScore < 0 ? 'bg-fids-red' : 'bg-fids-green'}`}
                          style={{
                            left: selectedFlight.sensorScore < 0 ? `${20 - (Math.abs(selectedFlight.sensorScore) * 80)}%` : '20%',
                            width: selectedFlight.sensorScore < 0 ? `${(Math.abs(selectedFlight.sensorScore) * 80)}%` : `${selectedFlight.sensorScore * 80}%`
                          }}
                        ></div>
                        <div className="absolute inset-0 flex items-center px-4 font-bold text-xs pointer-events-none drop-shadow-sm text-white">
                          Score: {selectedFlight.sensorScore.toFixed(3)}
                        </div>
                      </>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center font-bold text-[10px] uppercase text-gray-600 tracking-widest pointer-events-none">
                        [ AWAITING SENSOR ANOMALY EXECUTION ]
                      </div>
                    )}
                  </div>
                </div>

                {/* SVG Telemetry Profiler */}
                <div className="mt-6 pt-4 border-t border-[#222]">
                  <div className="text-gray-500 text-[10px] mb-2 uppercase tracking-widest flex justify-between drop-shadow-md">
                    <span>Engine Sensor Signature (Real-Time Parameter Graph)</span>
                    <span className="text-gray-600">Dynamic Z-Score Line Tracking</span>
                  </div>
                  <div className="w-full h-32 relative bg-[#0a0a0a] border border-[#333] rounded overflow-hidden">
                    {/* Background Grid Lines */}
                    <div className="absolute inset-0 flex flex-col justify-between pointer-events-none opacity-20 py-1">
                      <div className="h-[1px] w-full bg-gray-500"></div>
                      <div className="h-[1px] w-full bg-gray-500"></div>
                      <div className="h-[1px] w-full bg-fids-yellow relative"><span className="absolute -top-3 right-2 text-[8px] text-fids-yellow font-bold">Standard Baseline (0.0)</span></div>
                      <div className="h-[1px] w-full bg-gray-500"></div>
                      <div className="h-[1px] w-full bg-gray-500"></div>
                    </div>

                    {Object.keys(sensorFeatures).length > 0 ? (
                      <svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="w-full h-full relative z-10 overflow-visible px-2 py-1">
                        <path d={generateLinePath(sensorFeatures)} fill="none" stroke="#a855f7" strokeWidth="3" vectorEffect="non-scaling-stroke" className="drop-shadow-[0_0_5px_rgba(168,85,247,0.8)]" />
                        {/* Dots */}
                        {Object.keys(sensorFeatures).map((k, i) => {
                          const x = (i / (Object.keys(sensorFeatures).length - 1)) * 1000;
                          let val = Number(sensorFeatures[k]) || 0;
                          let y = 50 - (val * 12.5);
                          y = Math.max(2, Math.min(98, y));
                          return (
                            <g key={k}>
                              <circle cx={x} cy={y} r="3" fill="#e9d5ff" vectorEffect="non-scaling-stroke" />
                              {/* Display text name slightly above dot, only generic approximation so it doesn't clutter */}
                              <text x={x} y={y - 8} fill="#a855f7" fontSize="8" textAnchor="middle" className="font-mono tracking-widest pointer-events-none opacity-60 mix-blend-screen">{k}</text>
                            </g>
                          )
                        })}
                      </svg>
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center font-bold text-[10px] uppercase text-gray-600 tracking-widest pointer-events-none">
                        [ NO SENSOR DATA LINKED ]
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex flex-col lg:flex-row gap-6 mb-8">
                {/* Left Column: Delay Metrics */}
                <div className="flex-1 bg-[#0a0a0a] p-5 border border-[#333]">
                  <div className="flex items-center gap-3 mb-6 border-b border-[#333] pb-3">
                    <div className="bg-blue-900 text-blue-300 text-xs font-bold px-2 py-1 uppercase tracking-wider border border-blue-700">Random Forest</div>
                    <h3 className="text-lg text-gray-300 uppercase tracking-widest font-bold">FLIGHT WEATHER MATRIX</h3>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-y-5 gap-x-4">
                    {Object.keys(modelFeatures).map(key => {
                      const isEditable = typeof modelFeatures[key] === 'number';
                      return (
                        <div key={key} className={`flex flex-col ${isEditable ? '' : 'opacity-50'}`}>
                          <label className="text-gray-500 text-[10px] mb-1 uppercase tracking-widest">{key.replace(/_/g, ' ')}</label>
                          <input
                            type={isEditable ? "number" : "text"}
                            step="0.01"
                            disabled={!isEditable}
                            className="bg-black border border-[#444] text-white p-2 focus:outline-none focus:border-blue-500 font-mono text-sm transition-colors w-full"
                            value={modelFeatures[key]}
                            onChange={(e) => handleFeatureChange(key, e.target.value)}
                          />
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Right Column: Sensor Metrics */}
                <div className="flex-1 bg-[#0a0a0a] p-5 border border-[#333]">
                  <div className="flex items-center gap-3 mb-6 border-b border-[#333] pb-3">
                    <div className="bg-purple-900 text-purple-300 text-xs font-bold px-2 py-1 uppercase tracking-wider border border-purple-700">Isolation Forest</div>
                    <h3 className="text-lg text-gray-300 uppercase tracking-widest font-bold">ENGINE SENSOR TELEMETRY</h3>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-3 gap-y-5 gap-x-4">
                    {Object.keys(sensorFeatures).length > 0 ? (
                      Object.keys(sensorFeatures).map(key => (
                        <div key={key} className="flex flex-col">
                          <label className="text-gray-500 text-[9px] mb-1 tracking-widest h-6 flex items-end" title={NASA_PROBES[key] || "Telemetry"}>
                            {key}: {NASA_PROBES[key] || "Telemetry"}
                          </label>
                          <div className="relative">
                            <input
                              type="number" step="0.01"
                              className="bg-black border border-[#444] text-white p-2 focus:outline-none focus:border-purple-500 font-mono text-sm transition-colors w-full z-10 relative bg-transparent"
                              value={sensorFeatures[key]}
                              onChange={(e) => handleSensorChange(key, e.target.value)}
                            />
                            {/* Visual Indicator for standard deviations (-2 to 2) */}
                            <div className="absolute bottom-0 left-0 h-1 bg-purple-900/50 w-full"></div>
                            <div
                              className={`absolute bottom-0 left-0 h-1 transition-all ${Math.abs(sensorFeatures[key]) > 1.5 ? 'bg-fids-red' : 'bg-purple-500'}`}
                              style={{ width: `${Math.min(100, Math.max(0, ((Number(sensorFeatures[key]) + 3) / 6) * 100))}%` }}
                            ></div>
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="col-span-3 text-gray-600 text-center py-8 italic tracking-widest uppercase">
                        NO SENSOR TELEMETRY DATA LINKED TO THIS AIRCRAFT
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex justify-center mt-4 pt-6 pb-2 border-t-2 border-[#333] shrink-0">
                <button
                  onClick={handlePredictAll}
                  disabled={isPredictingDelay || isPredictingSensor}
                  className="bg-fids-yellow text-black hover:bg-yellow-400 hover:shadow-[0_0_20px_rgba(234,179,8,0.5)] font-bold text-xl px-12 py-4 rounded-sm uppercase tracking-widest transition-all disabled:opacity-50 border-2 border-fids-yellow"
                >
                  {(isPredictingDelay || isPredictingSensor) ? 'CALCULATING PROBABILITIES...' : 'EXECUTE PREDICTION AI'}
                </button>
              </div>

            </div>
          </div>
        </div>
      )}

      {/* Global AI Architecture Overview Modal */}
      {showArchitecture && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 modal-overlay p-4 md:p-8">
          <div className="fids-board bg-[#0a0a0a] border-2 border-[#444] p-6 lg:p-10 max-w-[95vw] lg:max-w-6xl w-full relative modal-content overflow-hidden max-h-[90vh] shadow-[0_0_50px_rgba(0,0,0,0.9)] flex flex-col">
            <button
              className="absolute top-4 right-6 text-4xl text-gray-500 hover:text-white z-20"
              onClick={() => setShowArchitecture(false)}
            >
              ×
            </button>
            <h2 className="text-white text-3xl font-bold border-b border-[#333] pb-4 mb-6 uppercase tracking-widest flex-none pr-10">
              Aviation Machine Learning Architecture
            </h2>

            <div className="flex flex-col gap-8 overflow-y-auto custom-scrollbar pr-4 flex-1">
              {/* Subsystem 1 */}
              <div className="border border-[#222] p-6 relative bg-black/50">
                <div className="absolute -top-3 left-4 bg-[#0a0a0a] px-2 text-fids-yellow text-sm tracking-widest uppercase font-bold">Subsystem 1</div>
                <h3 className="text-2xl font-bold text-gray-200 uppercase mb-2">Flight Delay Prediction Engine</h3>
                <div className="flex flex-col xl:flex-row gap-6 mt-4">
                  <div className="flex-1">
                    <p className="text-gray-400 mb-4 leading-relaxed tracking-wide">A supervised classification architecture trained to ingest environmental conditions and historically scheduled performance metrics. Its exact goal is determining binary logic—whether the flight will experience a departure blockage exceeding exactly 15 minutes.</p>
                    <p className="text-gray-400 mb-2 leading-relaxed tracking-wide"><strong>Methodology:</strong> By utilizing Ensemble Learning (Random Forests), the model aggregates the votes of 100 deep decision trees. It automatically isolates chaotic variable overlaps—for instance, understanding that high &lt;precipitation&gt; and high &lt;windspeed&gt; combined with a highly congested &lt;origin_airport&gt; creates exponential delay probability.</p>
                    <ul className="text-sm text-gray-300 space-y-2 mt-6 p-4 bg-[#111] border border-[#222]">
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-blue-500 rounded-full"></div> <span className="text-blue-400 w-32 inline-block font-bold">ALGORITHM:</span> Scikit-Learn RandomForestClassifier (100 Trees, Max Depth 15)</li>
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-blue-500 rounded-full"></div> <span className="text-blue-400 w-32 inline-block font-bold">ACCURACY:</span> ~85% Variance via Cross-Validation</li>
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-blue-500 rounded-full"></div> <span className="text-blue-400 w-32 inline-block font-bold">TARGET PIPELINE:</span> Probability Thresholds (Delay &gt; 15m vs Normal)</li>
                    </ul>
                  </div>
                  <div className="flex-1 bg-[#111] p-5 text-xs text-gray-400 border border-[#222] min-w-[300px]">
                    <div className="text-white mb-4 lg:mb-6 tracking-widest border-b border-[#333] pb-2 font-bold flex items-center gap-2"><div className="w-1.5 h-1.5 bg-fids-yellow rounded-full"></div> INGESTED WEATHER MATRIX</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-y-4 gap-x-2 font-mono">
                      <div><span className="text-blue-300 mr-2">str</span>carrier_code</div>
                      <div><span className="text-blue-300 mr-2">str</span>origin_airport</div>
                      <div><span className="text-blue-300 mr-2">str</span>dest_airport</div>
                      <div><span className="text-green-300 mr-2">int</span>scheduled_elapsed_time</div>
                      <div><span className="text-green-300 mr-2">flt</span>hourly_temperature</div>
                      <div><span className="text-green-300 mr-2">flt</span>hourly_precipitation</div>
                      <div><span className="text-green-300 mr-2">flt</span>hourly_visibility</div>
                      <div><span className="text-green-300 mr-2">flt</span>hourly_station_pressure</div>
                      <div><span className="text-green-300 mr-2">flt</span>hourly_windspeed</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Subsystem 2 */}
              <div className="border border-[#222] p-6 relative bg-black/50">
                <div className="absolute -top-3 left-4 bg-[#0a0a0a] px-2 text-fids-yellow text-sm tracking-widest uppercase font-bold">Subsystem 2</div>
                <h3 className="text-2xl font-bold text-gray-200 uppercase mb-2">Aircraft Engine Sensor Anomaly Detection</h3>
                <div className="flex flex-col xl:flex-row gap-6 mt-4">
                  <div className="flex-1">
                    <p className="text-gray-400 mb-4 leading-relaxed tracking-wide">An unsupervised mathematics model built for predictive maintenance. Because catastrophic engine failure is exceptionally rare, there is no symmetrical "labeled" dataset to train on. Instead, we use Isolation Forests to ingest high-density NASA C-MAPSS telemetry and mathematically cluster the definition of a "healthy" rotor profile.</p>
                    <p className="text-gray-400 mb-2 leading-relaxed tracking-wide"><strong>Methodology:</strong> The model recursively sections dimensional feature space. Normal engine profiles require extremely deep cuts to isolate, whereas true anomalies (massive sudden engine spikes) exist in sparse zones and are mathematically isolated immediately. If the isolation depth is suspiciously short, the system outputs a negative anomaly scalar.</p>
                    <ul className="text-sm text-gray-300 space-y-2 mt-6 p-4 bg-[#111] border border-[#222]">
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-purple-500 rounded-full"></div> <span className="text-purple-400 w-32 inline-block font-bold">ALGORITHM:</span> Scikit-Learn IsolationForest</li>
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-purple-500 rounded-full"></div> <span className="text-purple-400 w-32 inline-block font-bold">CONTAMINATION:</span> Set to strict 0.05 limit mathematically</li>
                      <li className="flex items-center gap-2"><div className="w-2 h-2 bg-purple-500 rounded-full"></div> <span className="text-purple-400 w-32 inline-block font-bold">TARGET PIPELINE:</span> Outlier Tagging (Score &lt; 0.0 is abnormal)</li>
                    </ul>
                  </div>
                  <div className="flex-1 bg-[#111] p-5 text-xs text-gray-400 border border-[#222] min-w-[300px]">
                    <div className="text-white mb-4 lg:mb-6 tracking-widest border-b border-[#333] pb-2 font-bold flex items-center gap-2"><div className="w-1.5 h-1.5 bg-fids-yellow rounded-full"></div> INGESTED NASA TELEMETRY ARRAY</div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-y-3 gap-x-2 font-mono">
                      {Object.keys(NASA_PROBES).map(key => (
                        <div key={key} className="flex gap-2">
                          <span className="text-purple-300 min-w-[24px]">[{key}]</span>
                          <span className="truncate" title={NASA_PROBES[key]}>{NASA_PROBES[key]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
