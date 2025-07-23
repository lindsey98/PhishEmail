/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT license.
 * See LICENSE in the project root for license information.
 */

/* global document, Office */

Office.onReady((info) => {
  if (info.host === Office.HostType.Outlook) {
    document.getElementById("sideload-msg").style.display = "none";
    document.getElementById("app-body").style.display = "flex";
    document.getElementById("run").onclick = run;
  }
});

function showExplanation(body, sender, recepient, subject) {
  const explanationButton = document.getElementById("detailed-explanation");
  const backButton = document.getElementById("return");
  const explanationBody = document.getElementById("html-body");
  explanationBody.innerHTML = body;
  explanationButton.onclick = () => {
    document.getElementById('tab-content').style.transform = 'translateX(-100%)';
    setTimeout(function() {
      window.scrollTo({top: 0, behavior: 'smooth'});
    }, 400);
  }
  backButton.onclick = () => {
    document.getElementById('tab-content').style.transform = 'translateX(0%)';
    setTimeout(function() {
      window.scrollTo({top: 0, behavior: 'smooth'});
    }, 400);
  }

  const senderInfo = document.getElementById("sender-info");
  const recepientInfo = document.getElementById("recepient-info");
  const subjectInfo = document.getElementById("subject");

  senderInfo.innerHTML = sender;
  recepientInfo.innerHTML = recepient;
  subjectInfo.innerHTML = subject;
}

function getActionExplanation(actions) {
  if (actions.length) {
    return `There are detected actions such as <em>"${actions[0]}"</em>, representing potentially malicious follow-up actions.`
  } else {
    return `There are no detected actions in the email.`
  }
}

function handleResponse(isInconsistent, matchedIdentity, identities, actions, sender) {
  const banner = document.getElementById("banner");
  if (isInconsistent) {
    banner.textContent = "Phishing Alert";
    banner.style.backgroundColor = "#FFEAC2";
  } else if (matchedIdentity != "No Matched Brand" && matchedIdentity != "No Prediction") {
    banner.textContent = "Likely Benign"; 
    banner.style.backgroundColor = "#c0ee95";
  } else {
    banner.textContent = "No Match"; 
    banner.style.backgroundColor = "#aee0f3";
  }

  const wrapper = document.getElementById("results-wrapper");
  const representedBrand = document.getElementById("represented-brand");
  const senderDomain = document.getElementById("sender-domain");
  const identityExplanation = document.getElementById("identity-explanation");
  const actionSection = document.getElementById("action");
  const actionExplanation = document.getElementById("action-explanation");
  const recommendation = document.getElementById("recommendation-explanation");
  senderDomain.textContent = sender;

  // handle phishing
  actionExplanation.innerHTML = getActionExplanation(actions);
  if (isInconsistent) {
    if (matchedIdentity.startsWith("Internal")) {
      let role = matchedIdentity.split("Internal:")[1];
      representedBrand.textContent = `Internal Role: ${role}`;
      identityExplanation.innerHTML = "The email claims to be from an internal role but uses an email address inconsistent with the recepient's domain, representing <strong>identity inconsistency.</strong>"
    } else {
      representedBrand.textContent = matchedIdentity;
      identityExplanation.innerHTML = "There is a <strong>detected mismatch</strong> between the represented brand & sender domain, representing <strong>identity inconsistency.</strong>"
    }
    recommendation.innerHTML = "Please exercise extreme caution and <strong>avoid opening</strong> any links or attachments in this email."
    // handle no matches
  } else if (matchedIdentity == "No Matched Brand") {
    representedBrand.textContent = identities[0];
    identityExplanation.innerHTML = `The email claims to be from <strong>${identities[0]}</strong>, but this does not match any known identity in our database.`
    recommendation.innerHTML = "As PiMRef is unable to verify the legitimacy of this email, <strong>exercise caution</strong> with any links or attachments included in this email."
  } else if (matchedIdentity == "No Prediction") {
    representedBrand.textContent = "Unidentifiable";
    identityExplanation.innerHTML = "Unfortunately, PiMRef is unable to identify any identities in this email.";
    recommendation.innerHTML = "As PiMRef is unable to verify the legitimacy of this email, <strong>exercise caution</strong> with any links or attachments included in this email."
    // handle benign: Consistent
  } else {
    if (matchedIdentity.startsWith("Consistent")) {
      actionSection.style.display = "none";
      if (matchedIdentity.startsWith("Consistent:")) { // internal consistent
        let role = matchedIdentity.split("Consistent:")[1]
        representedBrand.textContent = `Internal Role: ${role}`;
        identityExplanation.innerHTML = "This email claims to be from an internal role, and there is consistency between the sender and recepient's domain."
      } else {
        representedBrand.textContent = matchedIdentity;
        identityExplanation.innerHTML = `The email claims to be from ${matchedIdentity} and uses an email address which is consistent with the identity's domain.`
      }
      recommendation.innerHTML = "PiMRef has not detected any phishing characteristics in this email. Thus, you are advised to exercise miminal caution when opening any links or attachments included in this email."
      // handle benign: no action
    } else {
      if (matchedIdentity.startsWith("Internal")) {
        let role = matchedIdentity.split("Internal:")[1]
        representedBrand.textContent = `Internal Role: ${role}`;
        identityExplanation.innerHTML = "The email claims to be from an internal role but uses an email address inconsistent with the recepient's domain, representing <strong>identity inconsistency.</strong>"
      } else {
        representedBrand.textContent = matchedIdentity;
        identityExplanation.innerHTML = "There is a <strong>detected mismatch</strong> between the represented brand & sender domain, representing <strong>identity inconsistency.</strong>";
      }
      recommendation.innerHTML = "PiMRef has detected identity inconsistency in this email, but did not spot any signs of follow-up actions. However, do still exercise caution when <strong>opening any</strong> links or attachments in this email."
    }
  }
  wrapper.style.visibility = "visible";
  wrapper.style.opacity = "1";
}

// Package function into promise
function getAttachmentContentAsyncPromise(itemId, itemName) {
  return new Promise((resolve, reject) => {
      Office.context.mailbox.item.getAttachmentContentAsync(itemId, (asyncResult) => {
          if (asyncResult.status === Office.AsyncResultStatus.Succeeded) {
              resolve({'name': itemName, 'content': asyncResult.value.content});
          } else {
              reject(asyncResult.error);
          }
      });
  });
}

export async function run() {
  document.getElementById('run').style.display="none";
  document.getElementById('loading').style.visibility="visible";
  document.getElementById('loading').style.opacity="1";

  const item = Office.context.mailbox.item;
  // Get all image attachments from email. Note that this will not work for encrypted emails
  let attachments = item.attachments;
  let attachmentData = []
  let promises = []
  
  if (attachments && item.attachments.length > 0) {
    if (!attachments[0].id) {
      console.error('Email is Encrypted and Attachments cannot be accessed!');
    } else {
      for(let i = 0; i < attachments.length; i++) {
        let extension = attachments[i].name.split('.').pop();
        if (extension == 'png' || extension == 'jpg') {
          promises.push(getAttachmentContentAsyncPromise(attachments[i].id, attachments[i].name))
        }
      }
    }
  }
  
  // consolidate all the results, then proceed with the next step
  await Promise.all(promises).then((results) => {
    attachmentData = results
  })

  // send request to our server
  item.body.getAsync(Office.CoercionType.Html, function (result) {
    if (result.status === Office.AsyncResultStatus.Succeeded) {
      let email_data = {
        'subject': item.subject, 
        'sender': item.sender,
        'to': item.to,
        'bcc': item.bcc,
        'cc': item.cc,
        'body': result.value,
        'attachments': attachmentData
      }

      fetch('http://localhost:5000/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(email_data)
      })
      .then(response => { return response.json(); })
      .then(responseData => { 
        document.getElementById('loading').style.display="none";
        if (responseData.status == 'error') throw new Error(responseData.message)
        handleResponse(responseData.isInconsistent, responseData.matchedIdentity, responseData.identities, responseData.actions, item.sender.emailAddress);
        showExplanation(responseData.labelledHTML, responseData.labelledSender, responseData.labelledRecipient, responseData.labelledSubject);
      })
      .catch(error => {console.error('Error:', error); });
    } else {
      console.error("Error retrieving body HTML:", result.error.message);
    }
  });

  
}