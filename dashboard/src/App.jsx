import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { fetchRuns } from './utils/api';
import Sidebar from './components/Sidebar';
import MainView from './components/MainView';
import './index.css';

function App() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      setLoading(true);
      const data = await fetchRuns();
      // Sort by timestamp if available
      const sorted = [...data].sort((a, b) => b.id.localeCompare(a.id));
      setRuns(sorted);
      if (!selectedRun && sorted.length > 0) {
        setSelectedRun(sorted[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar
        runs={runs}
        selectedRun={selectedRun}
        onSelectRun={setSelectedRun}
        loading={loading}
      />
      <main className="main-content">
        {selectedRun ? (
          <MainView run={selectedRun} />
        ) : (
          <div className="empty-state" style={{
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-secondary)'
          }}>
            <Activity size={48} style={{ marginBottom: '16px', opacity: 0.5 }} />
            <p>Select a run to begin analysis</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
