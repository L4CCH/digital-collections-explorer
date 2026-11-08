import React from 'react';
import './Pagination.css';

const Pagination = ({ currentPage, setCurrentPage, hasMore, isLoading }) => {
  return (
    <div className="pagination">
      <button
        className="pagination-button"
        onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
        disabled={currentPage <= 1 || isLoading}
      >
        Previous
      </button>
      <span>Page {currentPage}</span>
      <button 
        className="pagination-button"
        onClick={() => setCurrentPage(prev => prev + 1)}
        disabled={!hasMore || isLoading}
      >
        Next
      </button>
    </div>
  );
};

export const ResultsPerPageDropdown = ({ resultsPerPage, setResultsPerPage, setCurrentPage, isLoading }) => {
  const handleResultsPerPageChange = (e) => {
    setResultsPerPage(parseInt(e.target.value, 10));
    setCurrentPage(1);
  };

  return (
    <div className="results-per-page">
      <label htmlFor="resultsPerPage">Results per page:</label>
      <select 
        id="resultsPerPage"
        value={resultsPerPage}
        onChange={handleResultsPerPageChange}
        className="results-dropdown"
        disabled={isLoading}
      >
        <option value={30}>30</option>
        <option value={50}>50</option>
        <option value={100}>100</option>
      </select>
    </div>
  );
};

export default Pagination;
