/**
 * DeepScript Frontend Utility Formatters
 * =====================================
 */

/**
 * Normalizes an uploaded filename or image path into a clean, human-readable basename.
 * Strips directory prefixes (both forward and backward slashes), handles concatenated
 * path artifacts, and eliminates duplicated file extensions.
 *
 * @param {string} raw - Filename or path string
 * @returns {string} - Clean basename (e.g. "faculty_tamil_brahmi.png")
 */
export function getCleanBasename(raw) {
  if (!raw || typeof raw !== 'string') return '';
  
  // 1. Remove query parameters or URL hash if present
  let clean = raw.split('?')[0].split('#')[0];
  
  // 2. Strip any directory path components (both / and \)
  clean = clean.split(/[/\\]/).pop().trim();
  
  // 3. Handle path concatenation artifacts (e.g. "experimentsfaculty_failurefaculty_tamil_brahmi.png")
  if (clean.includes('faculty_failurefaculty_')) {
    clean = clean.replace(/^.*faculty_failure/, '');
  } else if (clean.includes('experimentsfaculty_failure')) {
    clean = clean.replace(/^experimentsfaculty_failure/, '');
  }
  
  // 4. Eliminate duplicate file extensions (e.g. .png.png -> .png)
  clean = clean.replace(/(\.[a-z0-9]{3,4})\1+$/i, '$1');
  
  return clean;
}
