import { render, screen } from '@testing-library/react';
import App from './App';

// Axios ships as an ES module which Jest 27 does not parse by default.
// Mock it to avoid syntax errors during tests.
jest.mock('axios', () => ({ default: {} }));
jest.mock('d3', () => ({}));

test('renders header', () => {
  Object.defineProperty(window.HTMLElement.prototype, "scrollIntoView", {writable: true, value: jest.fn()});
  render(<App />);
  expect(screen.getByText(/Quantum Chat/i)).toBeTruthy();
});
