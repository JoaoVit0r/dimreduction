/***************************************************************************/
/*** Interactive Graphic Environment for Dimensionality Reduction        ***/
/***                                                                     ***/
/*** Copyright (C) 2006  David Corrêa Martins Junior                     ***/
/***                     Fabrício Martins Lopes                          ***/
/***                     Roberto Marcondes Cesar Junior                  ***/
/***                                                                     ***/
/*** This library is free software; you can redistribute it and/or       ***/
/*** modify it under the terms of the GNU Lesser General Public          ***/
/*** License as published by the Free Software Foundation; either        ***/
/*** version 2.1 of the License, or (at your option) any later version.  ***/
/***                                                                     ***/
/*** This library is distributed in the hope that it will be useful,     ***/
/*** but WITHOUT ANY WARRANTY; without even the implied warranty of      ***/
/*** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   ***/
/*** Lesser General Public License for more details.                     ***/
/***                                                                     ***/
/*** You should have received a copy of the GNU Lesser General Public    ***/
/*** License along with this library; if not, write to the Free Software ***/
/*** Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA       ***/
/*** 02110-1301  USA                                                     ***/
/***                                                                     ***/
/*** Contact: David Corr�a Martins Junior - davidjr@vision.ime.usp.br    ***/
/***          Fabr�cio Martins Lopes - fabriciolopes@vision.ime.usp.br   ***/
/***          Roberto Marcondes Cesar Junior - cesar@vision.ime.usp.br   ***/
/***************************************************************************/

/***************************************************************************/
/*** This class implements errors exceptions treatment.                  ***/
/***************************************************************************/

package fs;

import javax.swing.JOptionPane;

import utilities.IOFile;

/**
 * Custom exception class for handling feature selection and related errors.
 * <p>
 * This exception class extends RuntimeException and provides additional functionality
 * for handling application-specific errors in the feature selection and network
 * inference processes. It can optionally display error messages in a dialog box
 * and terminate the application if needed.
 * <p>
 * The class provides different constructors to handle various error scenarios
 * and supports both GUI and CLI error reporting modes.
 * 
 * @see RuntimeException
 */
public class FSException extends RuntimeException {
    
    /**
     * Default constructor for FSException.
     * Creates an exception with no message or cause.
     */
    public FSException() {
    }
    
    /**
     * Creates an exception with the specified error message and exit behavior.
     * <p>
     * If the exit parameter is true, this constructor will attempt to display
     * the error message in a dialog box and then terminate the application.
     * 
     * @param msg The detailed error message describing what went wrong.
     * @param exit If true, displays an error dialog and terminates the application.
     */
    public FSException(String msg, boolean exit) {
        super(msg);
        IOFile.PrintlnAndLog(msg, IOFile.VERBOSE_ERROR);
        if (exit)
            System.exit(9);
    }
    /**
     * Creates an exception with the specified error message.
     * <p>
     * This constructor displays the error message in a dialog box but does not
     * terminate the application.
     * 
     * @param msg The detailed error message describing what went wrong.
     */
    public FSException(String msg) {
        super(msg);
        IOFile.PrintlnAndLog(msg, IOFile.VERBOSE_ERROR);
    }
}